"""
cka_layer_mapping.py

Implements:
  1. Minibatch CKA (centred kernel alignment) between layer activations
     — uses the unbiased HSIC1 estimator so results are batch-size independent.
  2. Dynamic-programming layer mapping that maximises total CKA similarity
     subject to a maximum-offset constraint (Algorithm 1 from the paper).

Fixes applied vs. original:
  - BUG FIX: batches_run initialised before the collection loop.
  - BUG FIX: cka_layer_mapping handles lo > ln (source deeper than target)
    by transposing S, running the DP, then inverting the mapping.

Usage
-----
    from cka_layer_mapping import compute_cka_matrix, cka_layer_mapping

    S = compute_cka_matrix(
        original_model, new_model, dataloader,
        n_batches=32, device="cuda"
    )
    layer_dict = cka_layer_mapping(S, max_offset=2)
    # layer_dict[i] = j  means original layer i maps to new layer j
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HSIC / CKA kernels
# ---------------------------------------------------------------------------

def hsic1_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Unbiased estimator of HSIC (HSIC1) from Song et al. / Nguyen et al.

    K, L: (n, n) Gram matrices (diagonal should be zeroed before calling)

    HSIC1(K, L) = 1/(n(n-3)) * [tr(K̃L̃) + (1^T K̃ 1)(1^T L̃ 1)/(n-1)(n-2)
                                  - 2/(n-2) * 1^T K̃ L̃ 1]
    where K̃ = K with diagonal set to 0.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 samples for unbiased HSIC1.")

    K = K.clone()
    L = L.clone()
    K.fill_diagonal_(0.0)
    L.fill_diagonal_(0.0)

    ones = torch.ones(n, 1, device=K.device, dtype=K.dtype)

    tr_KL  = torch.trace(K @ L)
    sum_K  = (ones.T @ K @ ones).squeeze()
    sum_L  = (ones.T @ L @ ones).squeeze()
    KL_row = ones.T @ K @ L @ ones

    hsic = (
        tr_KL
        + sum_K * sum_L / ((n - 1) * (n - 2))
        - 2 * KL_row / (n - 2)
    ) / (n * (n - 3))
    return hsic


def cka_minibatch(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CKA between two activation matrices X and Y for a single minibatch.

    X: (n, d1)  activations from layer A
    Y: (n, d2)  activations from layer B

    Returns a scalar CKA value in [0, 1].
    """
    X = X.reshape(X.shape[0], -1).float()
    Y = Y.reshape(Y.shape[0], -1).float()

    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = hsic1_unbiased(K, L)
    hsic_kk = hsic1_unbiased(K, K)
    hsic_ll = hsic1_unbiased(L, L)

    denom = (hsic_kk * hsic_ll).sqrt()
    if denom < 1e-10:
        return torch.tensor(0.0, device=X.device)

    return hsic_kl / denom


# ---------------------------------------------------------------------------
# Hook-based activation collection
# ---------------------------------------------------------------------------

class ActivationCollector:
    """Registers forward hooks on a list of modules and accumulates activations."""

    def __init__(self, modules: list[nn.Module]):
        self.activations: dict[int, list[torch.Tensor]] = {
            i: [] for i in range(len(modules))
        }
        self._hooks = []
        for i, mod in enumerate(modules):
            hook = mod.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out.dim() == 3:
                out = out.mean(dim=1)
            self.activations[idx].append(out.detach().cpu())
        return hook

    def get_activations(self, idx: int) -> torch.Tensor:
        return torch.cat(self.activations[idx], dim=0)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def _get_transformer_layers(model: PreTrainedModel) -> list[nn.Module]:
    """
    Return the list of transformer block modules in order.
    Covers LLaMA/Mistral/Phi/Qwen style and GPT-NeoX style.
    """
    try:
        return list(model.model.layers)
    except AttributeError:
        pass
    try:
        return list(model.transformer.h)
    except AttributeError:
        pass
    try:
        return list(model.gpt_neox.layers)
    except AttributeError:
        pass
    raise AttributeError(
        "Cannot find transformer layers automatically. "
        "Override _get_transformer_layers() for your model family."
    )


# ---------------------------------------------------------------------------
# Full CKA similarity matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cka_matrix(
    original_model: PreTrainedModel,
    new_model: PreTrainedModel,
    dataloader: DataLoader,
    n_batches: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Build the CKA similarity matrix S ∈ R^{L_orig x L_new}.
    S[i, j] = CKA between layer i of original_model and layer j of new_model,
              averaged over n_batches minibatches.
    """
    original_model = original_model.to(device).eval()
    new_model      = new_model.to(device).eval()

    layers_orig = _get_transformer_layers(original_model)
    layers_new  = _get_transformer_layers(new_model)
    L_orig = len(layers_orig)
    L_new  = len(layers_new)
    logger.info(f"Layer depths — orig: {L_orig}, new: {L_new}")

    S = torch.zeros(L_orig, L_new)

    batch_orig_acts: list[dict[int, torch.Tensor]] = []
    batch_new_acts:  list[dict[int, torch.Tensor]] = []

    temp_orig: dict[int, torch.Tensor] = {}
    temp_new:  dict[int, torch.Tensor] = {}

    def make_hook_batch(store, idx):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() == 3:
                out = out.mean(dim=1)
            store[idx] = out.detach().cpu()
        return hook

    hooks = []
    for i, layer in enumerate(layers_orig):
        hooks.append(layer.register_forward_hook(make_hook_batch(temp_orig, i)))
    for j, layer in enumerate(layers_new):
        hooks.append(layer.register_forward_hook(make_hook_batch(temp_new, j)))

    batches_run = 0   # FIX: initialise before the loop
    with torch.no_grad():
        for batch in dataloader:
            if batches_run >= n_batches:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            temp_orig.clear()
            temp_new.clear()
            original_model(input_ids=input_ids, attention_mask=attention_mask)
            new_model(input_ids=input_ids, attention_mask=attention_mask)

            batch_orig_acts.append({k: v.clone() for k, v in temp_orig.items()})
            batch_new_acts.append({k: v.clone() for k, v in temp_new.items()})
            batches_run += 1

    for h in hooks:
        h.remove()

    logger.info(f"Collected activations over {batches_run} batches. Computing CKA matrix...")
    for i in range(L_orig):
        for j in range(L_new):
            cka_vals = []
            for b in range(batches_run):
                X   = batch_orig_acts[b][i]
                Y   = batch_new_acts[b][j]
                val = cka_minibatch(X, Y)
                cka_vals.append(val.item())
            S[i, j] = sum(cka_vals) / len(cka_vals)
        if (i + 1) % 4 == 0:
            logger.info(f"  CKA: processed {i+1}/{L_orig} original layers")

    return S


# ---------------------------------------------------------------------------
# Dynamic-programming layer mapping (Algorithm 1 from the paper)
# ---------------------------------------------------------------------------

def cka_layer_mapping(
    S: torch.Tensor,
    max_offset: Optional[int] = None,
) -> dict[int, int]:
    """
    Given a CKA similarity matrix S ∈ R^{L_orig x L_new}, find the mapping
    orig_layer_i -> new_layer_j that maximises the total CKA similarity while
    preserving layer order and staying within `max_offset` of the diagonal.

    Handles both directions:
      - L_orig <= L_new (source shallower): standard DP, all source layers mapped.
      - L_orig >  L_new (source deeper):   transposes S, runs DP on (L_new, L_orig),
        selects the L_new best-matched source layers, then inverts to orig->new.

    Args:
        S:           (L_orig, L_new) similarity matrix.
        max_offset:  Maximum allowed offset. Defaults to |L_new - L_orig| + 1.

    Returns:
        dict mapping orig_layer_i -> new_layer_j  (strictly increasing values).
        When L_orig > L_new, only L_new source layers appear as keys
        (the ones selected by the DP as best-matched).
    """
    S = S.float()

    # Sanitize NaN/Inf before anything else.
    # NaN arises in cross-family CKA when activation norms are near-zero for
    # some (i,j) pairs, causing 0/0 in the CKA denominator.  NaN > NEG_INF
    # evaluates to False in Python, so any NaN cell silently blocks all DP
    # paths through it — causing the "No valid mapping found" RuntimeError.
    # Replacing with 0.0 treats the cell as low-similarity rather than fatal.
    nan_count = torch.isnan(S).sum().item()
    inf_count = torch.isinf(S).sum().item()
    if nan_count > 0 or inf_count > 0:
        logger.warning(
            f"CKA matrix has {nan_count} NaN and {inf_count} Inf entries — "
            f"replacing with 0.0. This is expected for cross-family pairs "
            f"(different hidden sizes / tokenisers) but indicates high "
            f"architectural dissimilarity."
        )
        S = torch.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)

    lo, ln = S.shape

    # FIX: when source is deeper than target, transpose so DP always has lo <= ln.
    # The DP requires more columns than rows (it maps each row to a unique column).
    # After finding the mapping on the transposed matrix (new->orig), we invert it.
    transposed = lo > ln
    if transposed:
        S  = S.T
        lo, ln = ln, lo   # now lo <= ln guaranteed

    if max_offset is None:
        max_offset = abs(ln - lo) + 1

    NEG_INF = float("-inf")
    dp   = [[NEG_INF] * ln for _ in range(lo)]
    prev = [[-1]      * ln for _ in range(lo)]

    # Base case: first layer can map to columns 0..max_offset
    for j in range(min(max_offset + 1, ln)):
        dp[0][j] = S[0, j].item()

    # Fill DP
    for i in range(1, lo):
        j_min = i
        j_max = min(i + max_offset, ln - 1)
        for j in range(j_min, j_max + 1):
            best_val = NEG_INF
            best_k   = -1
            for k in range(max(0, j - max_offset - 1), j):
                if dp[i - 1][k] > best_val:
                    best_val = dp[i - 1][k]
                    best_k   = k
            if best_val > NEG_INF:
                dp[i][j]   = best_val + S[i, j].item()
                prev[i][j] = best_k

    last_row     = dp[lo - 1]
    best_final_j = max(range(ln), key=lambda j: last_row[j])

    if last_row[best_final_j] == NEG_INF:
        raise RuntimeError(
            "No valid layer mapping found. "
            "Try increasing max_offset or check S dimensions."
        )

    # Backtrack to recover the full path
    # In the (possibly transposed) DP:
    #   rows i ∈ 0..lo-1  correspond to the SHALLOWER model's layers
    #   cols j ∈ 0..ln-1  correspond to the DEEPER   model's layers
    forward_path = {}   # shallow_layer_i -> deep_layer_j
    j = best_final_j
    for i in range(lo - 1, -1, -1):
        forward_path[i] = j
        j = prev[i][j]

    logger.info(f"Total CKA score: {last_row[best_final_j]:.4f}")

    if not transposed:
        # S was NOT transposed: rows=orig (source), cols=new (target)
        # forward_path[orig_i] = new_j  →  this IS orig->new already
        result = forward_path
        logger.info(f"Layer mapping (orig→new): {result}")
    else:
        # S WAS transposed: orig had lo>ln so we flipped to (ln, lo)
        # After flip: rows=new/target (shallow, ln of original), cols=orig/source (deep, lo of original)
        # So forward_path[new_i] = orig_j
        # We need orig->new: for each (new_i, orig_j) pair, store orig_j -> new_i
        result = {}
        for new_i, orig_j in forward_path.items():
            result[orig_j] = new_i
        logger.info(f"Layer mapping (orig→new, via transpose+invert): {result}")

    # Sanity check: all values should be valid target indices
    tgt_depth = ln if not transposed else lo  # original lo before swap
    for orig_i, new_j in result.items():
        if new_j >= tgt_depth:
            logger.error(
                f"BAD MAPPING: orig layer {orig_i} → target layer {new_j} "
                f"but target only has {tgt_depth} layers!"
            )

    return result