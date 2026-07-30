"""
portable_peft.py

Main entry point. Subclasses PeftModel to add LoRASuite-style weight
portability across model architectures with 0 fine-tuning steps.

Architecture support: llama, neox (GPT-NeoX), bloom
"""

import torch
import logging
from pathlib import Path
from typing import Optional

from peft import PeftModel, PeftConfig
from peft.utils import set_peft_model_state_dict
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader

from embedding_transform import compute_embedding_transform
from cka_layer_mapping import compute_cka_matrix, cka_layer_mapping
from head_mapping import (
    get_attention_config,
    extract_layer_weights,
    extract_mlp_weights,
    compute_head_similarity_matrix,
    hungarian_head_mapping,
    transform_attention_lora,
    transform_fused_qkv_lora,
    transform_mlp_lora,
    make_lora_key_fn,
)

logger = logging.getLogger(__name__)

# Architectures where Q/K/V are fused into a single projection
FUSED_QKV_ARCHS = ("neox", "bloom")


def detect_arch(model: PreTrainedModel) -> str:
    """
    Detect architecture family from model class name.
    Returns one of: "llama", "neox", "bloom"
    """
    cls_name = type(model).__name__.lower()
    if "neox" in cls_name or "pythia" in cls_name:
        return "neox"
    if "bloom" in cls_name:
        return "bloom"
    # Covers LlamaForCausalLM, MistralForCausalLM, PhiForCausalLM,
    # Qwen2ForCausalLM, MiniCPMForCausalLM, etc.
    return "llama"


class PortablePeftModel(PeftModel):
    """
    PeftModel subclass that ports RL-trained LoRA weights from one backbone
    to another using LoRASuite-style algebraic transformations.

    Pipeline (all 0 fine-tuning steps):
        1. Compute embedding transform Wh
        2. Load or compute CKA similarity matrix
        3. Dynamic-programming layer mapping
        4. For each mapped layer pair:
           a. Hungarian head assignment
           b. Head-level LoRA transformation (Eq. 3)
           c. SVD back to low-rank matrices
           d. MLP transformation (direct A/B manipulation)
        5. Load ported weights into the new model
    """

    @classmethod
    def from_pretrained_ported(
        cls,
        new_model: PreTrainedModel,
        original_model: PreTrainedModel,
        lora_path: str,
        tokenizer_orig: Optional[PreTrainedTokenizer] = None,
        tokenizer_new: Optional[PreTrainedTokenizer] = None,
        calib_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        n_cka_batches: int = 32,
        max_layer_offset: Optional[int] = None,
        cka_matrix_path: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        arch_orig: Optional[str] = None,
        arch_new: Optional[str] = None,
        **peft_kwargs,
    ) -> "PortablePeftModel":
        """
        Port a LoRA adapter trained on `original_model` to `new_model`.

        Args:
            new_model:         Target backbone.
            original_model:    Source backbone the LoRA was trained on.
            lora_path:         Path to the saved PEFT adapter directory.
            tokenizer_orig:    Required when vocabulary sizes differ.
            tokenizer_new:     Required when vocabulary sizes differ.
            calib_loader:      Calibration DataLoader for CKA (can be None if
                               cka_matrix_path is provided and already exists).
            device:            Device for CKA computation ("cuda" or "cpu").
            n_cka_batches:     Minibatches to average for CKA estimation.
            max_layer_offset:  DP constraint for layer mapping.
            cka_matrix_path:   Path to cache/load a precomputed CKA matrix.
            dtype:             Float precision for transforms.
            arch_orig:         Force architecture ("llama"/"neox"/"bloom").
                               Auto-detected if None.
            arch_new:          Force architecture for new model. Auto-detected if None.
        """
        # ------------------------------------------------------------------ #
        # Step 0 — Load PEFT config + original LoRA weights                  #
        # ------------------------------------------------------------------ #
        config = PeftConfig.from_pretrained(lora_path)
        target_modules = list(config.target_modules)
        old_rank = config.r
        logger.info(f"LoRA rank={old_rank}, target_modules={target_modules}")

        orig_sd = _load_adapter_weights(lora_path)
        logger.info(f"Loaded {len(orig_sd)} original LoRA tensors.")

        _log_delta_norms(orig_sd, "original")

        # ------------------------------------------------------------------ #
        # Step 1 — Embedding transform Wh                                     #
        # ------------------------------------------------------------------ #
        logger.info("Computing Wh (embedding transform)...")
        Wh = compute_embedding_transform(
            original_model=original_model,
            new_model=new_model,
            tokenizer_orig=tokenizer_orig,
            tokenizer_new=tokenizer_new,
            device=device,
            dtype=dtype,
        )
        logger.info(f"Wh shape: {Wh.shape}")

        # Treat near-identity Wh as None to skip unnecessary multiplications
        if Wh.shape[0] == Wh.shape[1]:
            eye = torch.eye(Wh.shape[0], device=Wh.device, dtype=Wh.dtype)
            if torch.allclose(Wh, eye, atol=1e-4):
                logger.info("Wh is identity — hidden-size transform skipped.")
                Wh = None

        # Move Wh to CPU float32 for all downstream computation
        if Wh is not None:
            Wh = Wh.to(device="cpu", dtype=torch.float32)

        # ------------------------------------------------------------------ #
        # Step 2 — CKA similarity matrix                                      #
        # ------------------------------------------------------------------ #
        if cka_matrix_path and Path(cka_matrix_path).exists():
            logger.info(f"Loading precomputed CKA matrix from {cka_matrix_path}")
            S = torch.load(cka_matrix_path, map_location="cpu")
        else:
            if calib_loader is None:
                raise ValueError(
                    "Provide either calib_loader or an existing cka_matrix_path."
                )
            logger.info(f"Computing CKA matrix ({n_cka_batches} batches)...")
            S = compute_cka_matrix(
                original_model=original_model,
                new_model=new_model,
                dataloader=calib_loader,
                n_batches=n_cka_batches,
                device=device,
            )
            if cka_matrix_path:
                torch.save(S, cka_matrix_path)
                logger.info(f"Saved CKA matrix to {cka_matrix_path}")

        logger.info(f"CKA matrix: {S.shape}, mean={S.mean():.3f}, max={S.max():.3f}")

        # ------------------------------------------------------------------ #
        # Step 3 — Layer mapping                                              #
        # ------------------------------------------------------------------ #
        layer_map = cka_layer_mapping(S, max_offset=max_layer_offset)
        logger.info(f"Layer mapping (orig->new): {layer_map}")

        # ------------------------------------------------------------------ #
        # Step 4 — Transform LoRA weights layer by layer                      #
        # ------------------------------------------------------------------ #
        _arch_orig = arch_orig or detect_arch(original_model)
        _arch_new  = arch_new  or detect_arch(new_model)
        logger.info(f"Architectures: orig={_arch_orig}, new={_arch_new}")

        cfg_orig = get_attention_config(original_model)
        cfg_new  = get_attention_config(new_model)

        orig_state_dict = original_model.state_dict()
        new_state_dict  = new_model.state_dict()

        key_fn_orig = make_lora_key_fn(_arch_orig)
        key_fn_new  = make_lora_key_fn(_arch_new)

        ported_sd = {}

        for orig_idx, new_idx in layer_map.items():
            logger.info(f"  Layer {orig_idx} -> {new_idx}")

            # Extract base model weights for both layers
            orig_lw = extract_layer_weights(orig_state_dict, orig_idx, cfg_orig, _arch_orig)
            new_lw  = extract_layer_weights(new_state_dict,  new_idx,  cfg_new,  _arch_new)

            # Head similarity + Hungarian mapping
            sim = compute_head_similarity_matrix(orig_lw, new_lw, Wh)
            head_map = hungarian_head_mapping(sim)
            logger.debug(f"    Head mapping: {head_map}")

            # Attention LoRA transformation
            if _arch_orig in FUSED_QKV_ARCHS or _arch_new in FUSED_QKV_ARCHS:
                # If either side has fused QKV, use the fused variant
                attn_result = transform_fused_qkv_lora(
                    orig_lora=orig_sd,
                    orig_weights=orig_lw,
                    new_weights=new_lw,
                    head_mapping=head_map,
                    Wh=Wh,
                    old_rank=old_rank,
                    orig_layer_idx=orig_idx,
                    new_layer_idx=new_idx,
                    lora_key_fn_orig=key_fn_orig,
                    lora_key_fn_new=key_fn_new,
                )
            else:
                attn_result = transform_attention_lora(
                    orig_lora=orig_sd,
                    orig_weights=orig_lw,
                    new_weights=new_lw,
                    head_mapping=head_map,
                    Wh=Wh,
                    old_rank=old_rank,
                    orig_layer_idx=orig_idx,
                    new_layer_idx=new_idx,
                    lora_key_fn_orig=key_fn_orig,
                    lora_key_fn_new=key_fn_new,
                )
            ported_sd.update(attn_result)

            # MLP LoRA transformation
            orig_mlp = extract_mlp_weights(orig_state_dict, orig_idx, _arch_orig)
            new_mlp  = extract_mlp_weights(new_state_dict,  new_idx,  _arch_new)
            mlp_result = transform_mlp_lora(
                orig_lora=orig_sd,
                orig_mlp=orig_mlp,
                new_mlp=new_mlp,
                Wh=Wh,
                orig_layer_idx=orig_idx,
                new_layer_idx=new_idx,
                target_modules=target_modules,
                lora_key_fn_orig=key_fn_orig,
                lora_key_fn_new=key_fn_new,
            )
            ported_sd.update(mlp_result)

        logger.info(f"Ported {len(ported_sd)} tensors total.")
        _log_delta_norms(ported_sd, "ported")

        # ------------------------------------------------------------------ #
        # Step 5 — Attach ported weights to new model                         #
        # ------------------------------------------------------------------ #
        peft_model = cls(new_model, config, **peft_kwargs)
        result = set_peft_model_state_dict(peft_model, ported_sd, adapter_name="default")

        if hasattr(result, "unexpected_keys") and result.unexpected_keys:
            logger.warning(f"Unexpected keys: {result.unexpected_keys[:5]}")
        if hasattr(result, "missing_keys") and result.missing_keys:
            logger.warning(f"Missing keys: {result.missing_keys[:5]}")

        logger.info("PortablePeftModel ready (0 fine-tuning steps).")
        return peft_model

    def save_ported_adapter(self, save_path: str, **kwargs):
        self.save_pretrained(save_path, **kwargs)
        logger.info(f"Saved ported adapter to {save_path}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose_portability(
    original_sd: dict,
    ported_sd: dict,
    layer_map: dict,
    target_modules: list,
    arch_orig: str,
    arch_new: str,
) -> dict:
    """
    Per-module signal retention ratio: ||delta_W_ported|| / ||delta_W_orig||.
    Values close to 1.0 mean the transformation preserved the signal magnitude.
    Values << 1.0 suggest the policy signal is being attenuated.
    """
    key_fn_orig = make_lora_key_fn(arch_orig)
    key_fn_new  = make_lora_key_fn(arch_new)
    results = {}

    for orig_idx, new_idx in layer_map.items():
        for proj in ["q", "k", "v", "o"] + target_modules:
            kA_o = key_fn_orig(orig_idx, proj, "A")
            kB_o = key_fn_orig(orig_idx, proj, "B")
            kA_n = key_fn_new(new_idx,  proj, "A")
            kB_n = key_fn_new(new_idx,  proj, "B")
            if kA_o not in original_sd or kA_n not in ported_sd:
                continue

            norm_orig  = (original_sd[kB_o] @ original_sd[kA_o]).norm().item()
            norm_ported = (ported_sd[kB_n]  @ ported_sd[kA_n]).norm().item()
            ratio = norm_ported / (norm_orig + 1e-10)
            key = f"{orig_idx}->{new_idx}/{proj}"
            results[key] = ratio
            logger.info(f"  Signal retention {key}: {ratio:.4f}")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_adapter_weights(lora_path: str) -> dict:
    p = Path(lora_path)
    st = p / "adapter_model.safetensors"
    pt = p / "adapter_model.bin"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st), device="cpu")
    elif pt.exists():
        return torch.load(str(pt), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights in {lora_path}")


def _log_delta_norms(sd: dict, label: str):
    norms = []
    for k, v in sd.items():
        if "lora_B" in k:
            # find corresponding A
            key_a = k.replace("lora_B", "lora_A")
            if key_a in sd:
                norms.append((sd[k] @ sd[key_a]).norm().item())
    if norms:
        logger.info(
            f"  [{label}] delta_W norms — "
            f"mean={sum(norms)/len(norms):.4f}, "
            f"max={max(norms):.4f}, min={min(norms):.4f}"
        )
