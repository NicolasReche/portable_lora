"""
portability/run_portability.py

Run a single portability experiment: load a LoRA module (optionally porting
it to a different backbone), generate texts for a given dataset, and save
results to a JSON file.

Usage
-----
    # Verify adapter key format (always run this first for a new adapter)
    uv run python portability/run_portability.py --step verify \\
        --module ./models/sft_llama3.2_3b_sentiment_seed7097/checkpoint-200

    # Precompute CKA matrix for a specific model pair (once per pair)
    uv run python portability/run_portability.py --step cka \\
        --src_model llama32 --tgt_model qwen34b

    # Identity: evaluate original module on its own model (no porting)
    uv run python portability/run_portability.py --step generate \\
        --module ./models/sft_llama3.2_3b_sentiment_seed7097/checkpoint-200 \\
        --tgt_model llama32 \\
        --dataset yelp \\
        --attribute sentiment \\
        --output ./results/llama32_llama32_sentiment_yelp_seed7097.json

    # Port module from llama32 to qwen34b, then generate
    uv run python portability/run_portability.py --step generate \\
        --module ./models/sft_llama3.2_3b_sentiment_seed7097/checkpoint-200 \\
        --src_model llama32 \\
        --tgt_model qwen34b \\
        --dataset yelp \\
        --attribute sentiment \\
        --output ./results/llama32_to_qwen34b_sentiment_yelp_seed7097.json

    # Score an existing generation file
    uv run python portability/run_portability.py --step score \\
        --input  ./results/llama32_to_qwen34b_sentiment_yelp_seed7097.json \\
        --output ./results/llama32_to_qwen34b_sentiment_yelp_seed7097_metrics.json
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("portability")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "llama32": "meta-llama/Llama-3.2-3B",
    "llama31": "meta-llama/Llama-3.1-8B",
    "qwen34b": "Qwen/Qwen3-4B",
    "qwen38b": "Qwen/Qwen3-8B",
}

# max_offset for DP layer mapping
MAX_OFFSET = {
    ("llama32", "llama31"): 2, ("llama31", "llama32"): 2,
    ("qwen34b", "qwen38b"): 2, ("qwen38b", "qwen34b"): 2,
    ("llama32", "qwen34b"): 4, ("llama31", "qwen38b"): 4,
    ("qwen34b", "llama32"): 4, ("qwen38b", "llama31"): 4,
    ("llama32", "qwen38b"): 4, ("llama31", "qwen34b"): 4,
    ("qwen34b", "llama31"): 4, ("qwen38b", "llama32"): 4,
}

TRANSFER_CONDITION = {
    ("llama32", "llama31"): "same_family_diff_size",
    ("llama31", "llama32"): "same_family_diff_size",
    ("qwen34b", "qwen38b"): "same_family_diff_size",
    ("qwen38b", "qwen34b"): "same_family_diff_size",
    ("llama32", "qwen34b"): "cross_family_comparable_size",
    ("llama31", "qwen38b"): "cross_family_comparable_size",
    ("qwen34b", "llama32"): "cross_family_comparable_size",
    ("qwen38b", "llama31"): "cross_family_comparable_size",
    ("llama32", "qwen38b"): "cross_family_diff_size",
    ("llama31", "qwen34b"): "cross_family_diff_size",
    ("qwen34b", "llama31"): "cross_family_diff_size",
    ("qwen38b", "llama32"): "cross_family_diff_size",
}

# ---------------------------------------------------------------------------
# Dataset loaders — returns list of {"prompt": str, "attribute": str, "label": str}
# ---------------------------------------------------------------------------


def load_eval_prompts(
    dataset_name: str,
    attribute: str,
) -> list[dict]:
    """
    Load evaluation prompts from a dataset.
    Returns a list of {"prompt": str, "attribute": str, "label": str} dicts
    where `label` is the target label for CE scoring.

    For out-of-domain datasets (pplm, sts) labels are assigned
    by cycling through the attribute's label set, matching the
    paper's evaluation setup.
    """
    from datasets import load_dataset

    if attribute == "sentiment":
        labels    = ["Positive", "Negative"]
        tag       = "SENTIMENT"
    elif attribute == "topic":
        labels    = ["World", "Sports", "Business", "Science/Technology"]
        tag       = "TOPIC"
    else:
        raise ValueError(f"Unknown attribute: {attribute}. Use 'sentiment' or 'topic'.")

    records = []

    test_data = load_dataset("csv", data_files=dataset_name)['train']
    # iterate over test_data and create prompts
    for ex in test_data:
        for val in labels:
            prompt = f"[{tag}] {val} [\{tag}] [ANS] {ex['input']}".strip()
            records.append({"prompt": prompt, "attribute": attribute, "label": val})

    logger.info(f"Loaded {len(records)} prompts from {dataset_name} ({attribute})")
    return records


# ---------------------------------------------------------------------------
# Step 0: Verify adapter key format
# ---------------------------------------------------------------------------

def step_verify(module_path: str):
    from safetensors.torch import load_file

    p = Path(module_path)
    st = p / "adapter_model.safetensors"
    pt = p / "adapter_model.bin"
    if st.exists():
        sd = load_file(str(st))
    elif pt.exists():
        sd = torch.load(str(pt), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights in {module_path}")

    print(f"\nAdapter: {module_path}")
    print(f"Total keys: {len(sd)}")
    print("First 5 keys:")
    for k in list(sd.keys())[:5]:
        print(f"  {k}  shape={sd[k].shape}")
    has_default = any(".default." in k for k in sd)
    print(f"\nHas '.default.' suffix: {has_default}")
    if has_default:
        print("⚠ ACTION: Update make_lora_key_fn to include '.default.'")
    else:
        print("✓ Key format is correct for current make_lora_key_fn.")


# ---------------------------------------------------------------------------
# Step 1: Precompute CKA matrix for a specific model pair
# ---------------------------------------------------------------------------

def step_cka(
    src_model_key: str,
    tgt_model_key: str,
    cka_dir: str = "./cka_cache",
    n_batches: int = 32,
    batch_size: int = 8,
    device: str = "cuda",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from cka_layer_mapping import compute_cka_matrix

    # Canonical storage order: alphabetical
    a, b    = sorted([src_model_key, tgt_model_key])
    cka_path = _cka_path(src_model_key, tgt_model_key, cka_dir)

    if Path(cka_path).exists():
        logger.info(f"CKA already exists: {cka_path}")
        return cka_path

    os.makedirs(cka_dir, exist_ok=True)

    calib_texts = load_dataset("wikitext", "wikitext-2-raw-v1",
                               split="train")["text"]
    calib_texts = [t for t in calib_texts if len(t.strip()) > 20][:256]

    # Always compute with models in alphabetical order so the stored matrix
    # is unambiguously S[a_layers, b_layers] where a < b alphabetically.
    # _load_cka then transposes if the caller wants (b, a) direction.
    a, b = sorted([src_model_key, tgt_model_key])
    logger.info(f"Computing CKA: {a} (rows) ↔ {b} (cols)")

    tok = AutoTokenizer.from_pretrained(MODEL_IDS[a])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def collate(batch):
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]}

    loader = DataLoader(calib_texts, batch_size=batch_size,
                        collate_fn=collate, shuffle=False)

    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS[a], torch_dtype=torch.float16, device_map=device)
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS[b], torch_dtype=torch.float16, device_map=device)

    # S[i, j] = CKA(layer_i_of_a, layer_j_of_b)
    S = compute_cka_matrix(model_a, model_b, loader,
                            n_batches=n_batches, device=device)
    torch.save(S, cka_path)
    logger.info(f"Saved: {cka_path}  shape={S.shape}  "
                f"(rows={a}, cols={b})")

    del model_a, model_b
    torch.cuda.empty_cache()
    return cka_path


def _cka_path(src_key: str, tgt_key: str, cka_dir: str) -> str:
    a, b = sorted([src_key, tgt_key])
    return os.path.join(cka_dir, f"cka_{a}_{b}.pt")


def _load_cka(src_key: str, tgt_key: str, cka_dir: str) -> torch.Tensor:
    """
    Load CKA matrix for the (src → tgt) direction.

    Storage convention (enforced by step_cka):
        cka_{a}_{b}.pt  contains  S[a_layers, b_layers]
        where a < b alphabetically.

    If src==a and tgt==b: return S as-is  → shape (src_layers, tgt_layers) ✓
    If src==b and tgt==a: return S.T      → shape (src_layers, tgt_layers) ✓
    """
    path = _cka_path(src_key, tgt_key, cka_dir)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"CKA matrix not found: {path}\n"
            f"Run: --step cka --src_model {src_key} --tgt_model {tgt_key}"
        )
    S = torch.load(path, map_location="cpu")
    a, b = sorted([src_key, tgt_key])
    # S was stored as (a_layers, b_layers). If src==b, we need (b_layers, a_layers).
    if src_key == b:
        S = S.T
    logger.info(f"Loaded CKA {path}: shape={list(S.shape)} "
                f"(rows={src_key}, cols={tgt_key})")
    return S


# ---------------------------------------------------------------------------
# Step 2: Generate texts (with optional porting)
# ---------------------------------------------------------------------------


def get_ported_module(
    module_path: str,
    tgt_model_key: str,
    src_model_key: Optional[str] = None,
    cka_dir: str = "./cka_cache",
    device: str = "cuda",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    from peft.utils import set_peft_model_state_dict

    start_time_port = datetime.now(timezone.utc).isoformat()

    is_identity = (src_model_key is None or src_model_key == tgt_model_key)
    src_key     = tgt_model_key if is_identity else src_model_key

    logger.info(f"\n{'='*60}")
    logger.info(f"  {'Identity' if is_identity else 'Porting'}: "
                f"{src_key} → {tgt_model_key}")
    logger.info(f"{'='*60}")

    # ---- Load tokenizer for target model ----
    tok_tgt = AutoTokenizer.from_pretrained(MODEL_IDS[tgt_model_key])
    tok_tgt.padding_side = "left"
    if tok_tgt.pad_token is None:
        tok_tgt.pad_token = tok_tgt.eos_token

    if is_identity:
        # ---- Identity: load adapter directly on target model ----
        logger.info("Identity run — loading adapter directly...")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS[tgt_model_key],
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model = PeftModel.from_pretrained(base, module_path)
        model.eval()

    else:
        # ---- Porting run ----
        tok_src = AutoTokenizer.from_pretrained(MODEL_IDS[src_key])
        tok_src.padding_side = "left"
        if tok_src.pad_token is None:
            tok_src.pad_token = tok_src.eos_token

        logger.info("Loading backbones on CPU for algebraic transform...")
        src_model = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS[src_key], torch_dtype=torch.float32, device_map="cpu")
        tgt_model = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS[tgt_model_key], torch_dtype=torch.float32, device_map="cpu")

        # Import portability modules
        from embedding_transform import compute_embedding_transform
        from cka_layer_mapping import cka_layer_mapping
        from head_mapping import (
            get_attention_config, extract_layer_weights, extract_mlp_weights,
            compute_head_similarity_matrix, hungarian_head_mapping,
            transform_attention_lora, transform_mlp_lora, make_lora_key_fn,
        )
        from portable_peft import _load_adapter_weights, _log_delta_norms, detect_arch

        config         = PeftConfig.from_pretrained(module_path)
        orig_sd        = _load_adapter_weights(module_path)
        target_modules = list(config.target_modules)
        old_rank       = config.r
        logger.info(f"Adapter: rank={old_rank}, modules={target_modules}")

        # Wh
        logger.info("Computing Wh...")
        Wh = compute_embedding_transform(
            original_model=src_model, new_model=tgt_model,
            tokenizer_orig=tok_src, tokenizer_new=tok_tgt,
            device="cpu", dtype=torch.float32,
        )
        if Wh.shape[0] == Wh.shape[1] and \
                torch.allclose(Wh, torch.eye(Wh.shape[0]), atol=1e-4):
            logger.info("Wh ≈ identity — skipping.")
            Wh = None
        if Wh is not None:
            Wh = Wh.to(device="cpu", dtype=torch.float32)

        # CKA + layer mapping
        logger.info("Loading CKA and computing layer mapping...")
        S = _load_cka(src_key, tgt_model_key, cka_dir)
        lo, ln = S.shape
        # When source is deeper than target (lo > ln), the DP internally
        # transposes S and maps ln rows across lo columns. The offset must
        # be at least (lo - ln + 1) to guarantee a valid path exists.
        # We take the max of the configured offset and this minimum.
        configured_offset = MAX_OFFSET.get((src_key, tgt_model_key), 4)
        depth_diff        = abs(lo - ln)
        effective_offset  = max(configured_offset, depth_diff + 1)
        if effective_offset != configured_offset:
            logger.info(
                f"max_offset adjusted {configured_offset} → {effective_offset} "
                f"(layer depth difference: {depth_diff})"
            )
        layer_map = cka_layer_mapping(S, max_offset=effective_offset)
        logger.info(f"Layer mapping: {layer_map}")

        # Transform LoRA weights
        arch_src = detect_arch(src_model)
        arch_tgt = detect_arch(tgt_model)
        cfg_src  = get_attention_config(src_model)
        cfg_tgt  = get_attention_config(tgt_model)
        src_sd   = src_model.state_dict()
        tgt_sd   = tgt_model.state_dict()
        kfn_src  = make_lora_key_fn(arch_src)
        kfn_tgt  = make_lora_key_fn(arch_tgt)

        ported_sd = {}
        for orig_idx, new_idx in layer_map.items():
            orig_lw = extract_layer_weights(src_sd, orig_idx, cfg_src, arch_src)
            new_lw  = extract_layer_weights(tgt_sd, new_idx,  cfg_tgt, arch_tgt)

            sim      = compute_head_similarity_matrix(orig_lw, new_lw, Wh)
            head_map = hungarian_head_mapping(sim)

            ported_sd.update(transform_attention_lora(
                orig_lora=orig_sd, orig_weights=orig_lw, new_weights=new_lw,
                head_mapping=head_map, Wh=Wh, old_rank=old_rank,
                orig_layer_idx=orig_idx, new_layer_idx=new_idx,
                lora_key_fn_orig=kfn_src, lora_key_fn_new=kfn_tgt,
            ))
            ported_sd.update(transform_mlp_lora(
                orig_lora=orig_sd,
                orig_mlp=extract_mlp_weights(src_sd, orig_idx, arch_src),
                new_mlp=extract_mlp_weights(tgt_sd, new_idx,  arch_tgt),
                Wh=Wh, orig_layer_idx=orig_idx, new_layer_idx=new_idx,
                target_modules=target_modules,
                lora_key_fn_orig=kfn_src, lora_key_fn_new=kfn_tgt,
            ))

        _log_delta_norms(ported_sd, "ported")

        # Free source model
        del src_model, src_sd, orig_sd
        torch.cuda.empty_cache()

        # Attach ported weights to target model
        logger.info("Attaching ported weights...")
        peft_model_cpu = PeftModel(tgt_model, config)
        result = set_peft_model_state_dict(
            peft_model_cpu, ported_sd, adapter_name="default"
        )
        if hasattr(result, "missing_keys") and result.missing_keys:
            n_missing = len(result.missing_keys)
            coverage  = 1 - n_missing / max(len(ported_sd), 1)
            logger.warning(f"Key coverage: {coverage:.1%} "
                           f"({n_missing} missing)")

        # Move to GPU
        model = peft_model_cpu.to(device)
        model.eval()
        del peft_model_cpu, tgt_sd, ported_sd
        torch.cuda.empty_cache()
    end_time_port = datetime.now(timezone.utc).isoformat()

    return model, tok_tgt, start_time_port, end_time_port


def step_generate(
    module_path: str,
    tgt_model_key: str,
    dataset_names: List[str],
    attribute: str,
    output_path: str,
    src_model_key: Optional[str] = None,
    cka_dir: str = "./cka_cache",
    max_new_tokens: int = 100,
    batch_size: int = 16,
    device: str = "cuda",
):
    """
    Load a module (porting it if src_model_key != tgt_model_key),
    generate texts for the given dataset, and save to output_path.

    If src_model_key is None or equal to tgt_model_key: identity run (no porting).
    Otherwise: port the module algebraically from src to tgt before inference.

    Output JSON schema:
    {
      "start_time":  "...",
      "end_time":    "...",
      "model":       "meta-llama/Llama-3.2-3B",
      "module":      "./models/sft_llama3.2_3b_sentiment_seed7097/checkpoint-200",
      "src_model":   "llama32",
      "tgt_model":   "qwen34b",
      "condition":   "cross_family_comparable_size",  (or "identity")
      "dataset":     "yelp",
      "attribute":   "sentiment",
      "test_sets": {
        "test_data1": {
            "generated_texts": [
                {
                "prompt":         "[SENTIMENT] Positive [\\SENTIMENT] [ANS] The food was",
                "raw_completion": "absolutely fantastic! Best meal ever. [\\ANS]",
                "completion":     "absolutely fantastic! Best meal ever.",
                "label":      "Positive"
                }, ...
            ],
            "start_time": "...",
            "end_time":   "..."
        }
      }
    }
    """
    is_identity = (src_model_key is None or src_model_key == tgt_model_key)
    src_key     = tgt_model_key if is_identity else src_model_key
    condition   = "identity" if is_identity else \
                TRANSFER_CONDITION.get((src_key, tgt_model_key), "unknown")
    logger.info(f"\n{'='*60}")
    logger.info(f"  {'Identity' if is_identity else 'Porting'}: "
                f"{src_key} → {tgt_model_key}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"{'='*60}")
    model, tok_tgt, start_port, end_port = get_ported_module(
        module_path, tgt_model_key, src_model_key, cka_dir, device
    )
    output = {
        "start_time_port":     start_port,
        "end_time_port":       end_port,
        "model":          MODEL_IDS[tgt_model_key],
        "module":         str(module_path),
        "src_model":      src_key,
        "tgt_model":      tgt_model_key,
        "condition":      condition,
        "attribute":      attribute,
        "test_sets": {},
    }
    for dataset_name in dataset_names:
        start_time = datetime.now(timezone.utc).isoformat()
        logger.info(f"  Dataset: {dataset_name}  Attribute: {attribute}")

        # ---- Load prompts ----
        records = load_eval_prompts(dataset_name, attribute)
        prompts  = [r["prompt"]    for r in records]
        attrs    = [r["label"] for r in records]

        # ---- Generate ----
        logger.info(f"Generating {len(prompts)} completions...")
        generated_texts = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            batch_attrs   = attrs[i: i + batch_size]

            enc = tok_tgt(
                batch_prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=1024,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok_tgt.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4
                )

            for ids, prompt, attr in zip(out_ids, batch_prompts, batch_attrs):
                raw = tok_tgt.decode(ids[prompt_len:], skip_special_tokens=True)
                clean = raw.replace("[\ANS]", "").replace("[ANS]", "").strip()
                generated_texts.append({
                    "prompt":         prompt,
                    "raw_completion": raw,
                    "completion":     clean,
                    "label":      attr,
                })

            logger.info(f"  {min(i + batch_size, len(prompts))}/{len(prompts)} done")

        # ---- Free model ----
        #del model
        #torch.cuda.empty_cache()

        end_time = datetime.now(timezone.utc).isoformat()

        # ---- Save ----
        output["test_sets"][dataset_name] = {
            "generated_texts": generated_texts,
            "start_time": start_time,
            "end_time": end_time
        }

        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(generated_texts)} texts → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Step 3: Score a generation file
# ---------------------------------------------------------------------------

def step_score(
    input_path: str,
    output_path: str,
    device: str = "cuda",
    compute_slor: bool = True,
):
    """
    Load a generation JSON, compute CE / Distinct-n / SLOR,
    and save a metrics JSON alongside it.
    """
    from evaluation.evaluate import (
        ControlEffectivenessScorer, distinct_n, SLORScorer
    )

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    attribute = data["attribute"]
    texts     = [r["completion"] for r in data["generated_texts"]]
    targets   = [r["attribute"]  for r in data["generated_texts"]]

    logger.info(f"Scoring {len(texts)} texts ({attribute})...")

    ce_scorer = ControlEffectivenessScorer(attribute, device=device)
    ce        = ce_scorer.score(texts, targets)

    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)
    d3 = distinct_n(texts, 3)

    slor = None
    if compute_slor:
        logger.info("Computing SLOR (slow)...")
        slor = SLORScorer(device=device).score(texts)

    metrics = {
        "input_file":   input_path,
        "model":        data["model"],
        "module":       data["module"],
        "src_model":    data.get("src_model"),
        "tgt_model":    data.get("tgt_model"),
        "condition":    data.get("condition"),
        "dataset":      data["dataset"],
        "attribute":    attribute,
        "n_texts":      len(texts),
        "CE":           ce,
        "dist1":        d1,
        "dist2":        d2,
        "dist3":        d3,
        "SLOR":         slor,
    }

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved → {output_path}")
    logger.info(f"  CE={ce:.1f}  dist1={d1:.3f}  dist2={d2:.3f}  "
                f"dist3={d3:.3f}  SLOR={slor}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--step", required=True,
        choices=["verify", "cka", "generate", "score"],
        help=(
            "verify   — inspect adapter key format\n"
            "cka      — precompute CKA matrix for a model pair\n"
            "generate — port module and generate texts\n"
            "score    — compute metrics from a generation file"
        ),
    )

    # --- verify / generate ---
    parser.add_argument("--module",     type=str, default=None,
                        help="Path to the trained LoRA adapter directory")

    # --- generate ---
    parser.add_argument("--src_model",  type=str, default=None,
                        choices=list(MODEL_IDS.keys()),
                        help="Model the module was trained on. "
                             "Omit (or set equal to --tgt_model) for identity run.")
    parser.add_argument("--tgt_model",  type=str, default=None,
                        choices=list(MODEL_IDS.keys()),
                        help="Model to run inference on (after porting if needed).")
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                        help="List of test CSV paths to generate on after training.")
    parser.add_argument("--attribute",  type=str, default="sentiment",
                        choices=["sentiment", "topic"],
                        help="Control attribute.")
    parser.add_argument("--n_prompts",  type=int, default=100,
                        help="Number of prompts to generate.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output",     type=str, default=None,
                        help="Output JSON path for --step generate or score.")

    # --- cka ---
    parser.add_argument("--cka_dir",    type=str, default="./cka_cache")
    parser.add_argument("--n_cka_batches", type=int, default=32)

    # --- score ---
    parser.add_argument("--input",      type=str, default=None,
                        help="Generation JSON to score (--step score).")
    parser.add_argument("--no_slor",    action="store_true",
                        help="Skip SLOR computation.")

    # --- shared ---
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--seed",     type=int, default=7097)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # ---- verify ----
    if args.step == "verify":
        if not args.module:
            parser.error("--module required for --step verify")
        step_verify(args.module)

    # ---- cka ----
    elif args.step == "cka":
        if not args.src_model or not args.tgt_model:
            parser.error("--src_model and --tgt_model required for --step cka")
        step_cka(
            src_model_key=args.src_model,
            tgt_model_key=args.tgt_model,
            cka_dir=args.cka_dir,
            n_batches=args.n_cka_batches,
            device=args.device,
        )

    # ---- generate ----
    elif args.step == "generate":
        if not args.module:
            parser.error("--module required for --step generate")
        if not args.tgt_model:
            parser.error("--tgt_model required for --step generate")

        # Auto-build output filename if not specified
        output = args.output
        if output is None:
            src  = args.src_model or args.tgt_model
            name = f"{src}_to_{args.tgt_model}_{args.attribute}_{args.dataset[0]}.json"
            output = os.path.join("./results/portability", name)
            logger.info(f"No --output specified, using: {output}")

        step_generate(
            module_path=args.module,
            tgt_model_key=args.tgt_model,
            dataset_names=args.dataset,
            attribute=args.attribute,
            output_path=output,
            src_model_key=args.src_model,
            cka_dir=args.cka_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
        )

    # ---- score ----
    elif args.step == "score":
        if not args.input:
            parser.error("--input required for --step score")
        output = args.output or args.input.replace(".json", "_metrics.json")
        step_score(
            input_path=args.input,
            output_path=output,
            device=args.device,
            compute_slor=not args.no_slor,
        )
