"""
embedding_transform.py

Computes the hidden-size transformation matrix Wh from the embedding weights
of two models. This handles three scenarios:
  1. Only hidden size changes         (e.g. Phi-1.5 -> Phi-2)
  2. Only vocabulary size changes     (e.g. LLaMA-2-7B -> LLaMA-3-8B)
  3. Both change simultaneously       (e.g. MiniCPM-1B -> MiniCPM-2B)

Wh = E_o^{-1} E_n  (using shared-token intersection when vocab differs)
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_shared_token_ids(
    tokenizer_orig: PreTrainedTokenizer,
    tokenizer_new: PreTrainedTokenizer,
) -> tuple[list[int], list[int]]:
    """
    Find tokens that exist in both vocabularies.
    Returns (orig_ids, new_ids) — parallel lists of token indices
    for shared tokens, in the same order.
    """
    vocab_orig = tokenizer_orig.get_vocab()  # token_str -> id
    vocab_new = tokenizer_new.get_vocab()

    shared_tokens = set(vocab_orig.keys()) & set(vocab_new.keys())
    logger.info(
        f"Vocab sizes: orig={len(vocab_orig)}, new={len(vocab_new)}, "
        f"shared={len(shared_tokens)}"
    )

    if len(shared_tokens) < 100:
        raise ValueError(
            f"Too few shared tokens ({len(shared_tokens)}). "
            "Check that both tokenizers are for the same language/domain."
        )

    orig_ids = [vocab_orig[t] for t in shared_tokens]
    new_ids = [vocab_new[t] for t in shared_tokens]
    return orig_ids, new_ids


def compute_embedding_transform(
    original_model: PreTrainedModel,
    new_model: PreTrainedModel,
    tokenizer_orig: Optional[PreTrainedTokenizer] = None,
    tokenizer_new: Optional[PreTrainedTokenizer] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rcond: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the transformation matrix Wh such that:
        E_n ≈ E_o @ Wh
        => Wh = pinv(E_o) @ E_n

    When vocab sizes differ, only shared tokens are used.

    Args:
        original_model: The original backbone.
        new_model:       The new (upgraded) backbone.
        tokenizer_orig:  Required when vocab sizes differ.
        tokenizer_new:   Required when vocab sizes differ.
        device:          Where to run the computation.
        dtype:           Precision for the computation.
        rcond:           Cutoff ratio for small singular values in pinv.

    Returns:
        Wh: Tensor of shape (d_orig, d_new)
    """
    E_o = _get_embedding_matrix(original_model).to(device=device, dtype=dtype)
    E_n = _get_embedding_matrix(new_model).to(device=device, dtype=dtype)

    vocab_orig, d_orig = E_o.shape
    vocab_new, d_new = E_n.shape

    logger.info(f"Embedding shapes — orig: {E_o.shape}, new: {E_n.shape}")

    vocab_same = (vocab_orig == vocab_new)
    hidden_same = (d_orig == d_new)

    if vocab_same and hidden_same:
        logger.info("Vocab and hidden sizes are identical — Wh = Identity")
        return torch.eye(d_orig, device=device, dtype=dtype)

    if not vocab_same:
        # Need tokenizers to find the shared-token intersection
        if tokenizer_orig is None or tokenizer_new is None:
            raise ValueError(
                "Tokenizers are required when vocabulary sizes differ."
            )
        orig_ids, new_ids = get_shared_token_ids(tokenizer_orig, tokenizer_new)
        E_o_shared = E_o[orig_ids]   # (n_shared, d_orig)
        E_n_shared = E_n[new_ids]    # (n_shared, d_new)
    else:
        E_o_shared = E_o             # (vocab, d_orig)
        E_n_shared = E_n             # (vocab, d_new)

    # Wh = pinv(E_o_shared) @ E_n_shared   shape: (d_orig, d_new)
    # We use SVD-based pseudoinverse for numerical stability.
    Wh = torch.linalg.lstsq(
        E_o_shared, E_n_shared, rcond=rcond, driver="gelsd"
    ).solution

    # Sanity-check: reconstruction error on shared tokens
    recon_err = (E_o_shared @ Wh - E_n_shared).norm() / E_n_shared.norm()
    logger.info(f"Embedding reconstruction relative error: {recon_err:.4f}")
    if recon_err > 0.1:
        logger.warning(
            "High reconstruction error — the two models may be too "
            "architecturally dissimilar for a linear embedding transform."
        )

    return Wh  # (d_orig, d_new)


def compute_intermediate_transform(
    original_model: PreTrainedModel,
    new_model: PreTrainedModel,
    Wh: torch.Tensor,
    layer_idx: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute the intermediate-dimension transformation matrix Wi for
    up/down projection layers:

        Wi = W_o^{-1} @ Wh @ W_n

    where W_o and W_n are the up-projection weights of layer `layer_idx`
    in the original and new models respectively.

    Returns:
        Wi: Tensor of shape (d_int_orig, d_int_new)
    """
    Wo = _get_up_proj(original_model, layer_idx).to(device=device, dtype=dtype)
    Wn = _get_up_proj(new_model, layer_idx).to(device=device, dtype=dtype)
    Wh = Wh.to(device=device, dtype=dtype)

    # Wi = pinv(Wo) @ Wh @ Wn
    # Wo: (d_int_orig, d_hidden_orig)
    # Wh: (d_hidden_orig, d_hidden_new)
    # Wn: (d_int_new,  d_hidden_new)   => Wn.T: (d_hidden_new, d_int_new)
    middle = Wh @ Wn.T          # (d_hidden_orig, d_int_new)
    Wi = torch.linalg.lstsq(Wo, middle, driver="gelsd").solution
    return Wi  # (d_int_orig, d_int_new)


# ---------------------------------------------------------------------------
# Helpers — adjust these to match your model's module naming convention
# ---------------------------------------------------------------------------

def _get_embedding_matrix(model: PreTrainedModel) -> torch.Tensor:
    """Extract the input embedding weight from a HuggingFace model."""
    # Works for most decoder-only models (LLaMA, Mistral, Phi, GPT-NeoX…)
    for name in ("model.embed_tokens.weight",
                 "transformer.wte.weight",
                 "gpt_neox.embed_in.weight"):
        try:
            return _get_param(model, name).detach().clone()
        except AttributeError:
            continue
    raise AttributeError(
        "Could not find embedding matrix. "
        "Override _get_embedding_matrix() for your model family."
    )


def _get_up_proj(model: PreTrainedModel, layer_idx: int) -> torch.Tensor:
    """Extract the up-projection weight from a given layer."""
    # LLaMA / Mistral style
    try:
        return (
            model.model.layers[layer_idx]
            .mlp.up_proj.weight
            .detach().clone()
        )
    except AttributeError:
        pass
    # GPT-NeoX style
    try:
        return (
            model.gpt_neox.layers[layer_idx]
            .mlp.dense_h_to_4h.weight
            .detach().clone()
        )
    except AttributeError:
        pass
    raise AttributeError(
        f"Could not find up-projection at layer {layer_idx}. "
        "Override _get_up_proj() for your model family."
    )


def _get_param(model: torch.nn.Module, dotted_path: str) -> torch.nn.Parameter:
    obj = model
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj
