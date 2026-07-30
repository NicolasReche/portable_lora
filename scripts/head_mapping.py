"""
head_mapping.py

Implements attention head mapping and LoRA weight transformation.

Equation (3) from the paper (verified against reference implementation):
    For Q projection:
        W_Q_x  = W_old_Q[head_i] @ Wh @ W_new_Q[head_j].T
        ΔW_Q_new[head_j] = W_Q_x.T @ ΔW_Q_old[head_i] @ Wh

    For O projection (column-split):
        W_O_x  = W_old_O[:, head_i].T @ Wh @ W_new_O[:, head_j]
        ΔW_O_new[:, head_j] = Wh.T @ ΔW_O_old[:, head_i] @ W_O_x

Same pattern applies to K and V as Q.

MLP transforms avoid SVD and directly manipulate A/B:
    up_proj:   B_new = W_new_U @ pinv(Wh) @ pinv(W_old_U) @ B_old
               A_new = A_old @ Wh
    down_proj: B_new = pinv(Wh) @ B_old
               A_new = A_old @ pinv(W_old_D) @ Wh @ W_new_D
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from transformers import PreTrainedModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GQA utilities
# ---------------------------------------------------------------------------

def repeat_kv_heads(
    W: torch.Tensor,   # (n_kv_heads * head_dim, hidden_size)
    head_dim: int,
    n_rep: int,
) -> torch.Tensor:
    """Expand KV projection from n_kv_heads to n_kv_heads * n_rep heads."""
    if n_rep == 1:
        return W
    n_kv_heads = W.shape[0] // head_dim
    W = W.view(n_kv_heads, head_dim, W.shape[1])
    W = torch.repeat_interleave(W, repeats=n_rep, dim=0)
    return W.view(-1, W.shape[2])


def collapse_kv_heads(
    delta_W: torch.Tensor,   # (n_q_heads * head_dim, hidden_size)
    n_kv_heads: int,
    n_rep: int,
    head_dim: int,
) -> torch.Tensor:
    """Collapse expanded KV delta back to n_kv_heads by averaging groups."""
    if n_rep == 1:
        return delta_W
    hidden_size = delta_W.shape[1]
    delta_W = delta_W.view(n_kv_heads, n_rep, head_dim, hidden_size)
    return delta_W.mean(dim=1).view(n_kv_heads * head_dim, hidden_size)


# ---------------------------------------------------------------------------
# Attention config
# ---------------------------------------------------------------------------

class AttentionConfig:
    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int, hidden_size: int):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.n_rep = n_heads // n_kv_heads


def get_attention_config(model: PreTrainedModel) -> AttentionConfig:
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    hidden_size = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", None) or (hidden_size // n_heads)
    return AttentionConfig(n_heads, n_kv_heads, head_dim, hidden_size)


# ---------------------------------------------------------------------------
# Architecture-agnostic weight extraction
# ---------------------------------------------------------------------------

class LayerWeights:
    """
    Q/K/V/O matrices for one layer, with KV already expanded for GQA.
    Q/K/V: (n_heads * head_dim, hidden_size)
    O:     (hidden_size, n_heads * head_dim)
    """
    def __init__(self, Q, K, V, O, cfg: AttentionConfig):
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.cfg = cfg


def extract_layer_weights(
    model_state_dict: dict,
    layer_idx: int,
    cfg: AttentionConfig,
    arch: str,
) -> LayerWeights:
    """
    Extract Q/K/V/O weight matrices for a layer, expanding KV for GQA.
    arch: one of "llama", "neox", "bloom"
    """
    sd = model_state_dict
    h = layer_idx
    f32 = dict(device="cpu", dtype=torch.float32)

    if arch == "llama":
        Q = sd[f"model.layers.{h}.self_attn.q_proj.weight"].to(**f32)
        K = sd[f"model.layers.{h}.self_attn.k_proj.weight"].to(**f32)
        V = sd[f"model.layers.{h}.self_attn.v_proj.weight"].to(**f32)
        O = sd[f"model.layers.{h}.self_attn.o_proj.weight"].to(**f32)
        if cfg.n_rep > 1:
            K = repeat_kv_heads(K, cfg.head_dim, cfg.n_rep)
            V = repeat_kv_heads(V, cfg.head_dim, cfg.n_rep)

    elif arch in ("neox", "bloom"):
        # Q, K, V interleaved per head: layout [Q_h, K_h, V_h] for each head h
        if arch == "neox":
            QKV = sd[f"gpt_neox.layers.{h}.attention.query_key_value.weight"].to(**f32)
            O   = sd[f"gpt_neox.layers.{h}.attention.dense.weight"].to(**f32)
        else:
            QKV = sd[f"transformer.h.{h}.self_attention.query_key_value.weight"].to(**f32)
            O   = sd[f"transformer.h.{h}.self_attention.dense.weight"].to(**f32)
        Q = _deinterleave_qkv(QKV, cfg.n_heads, cfg.head_dim, 0)
        K = _deinterleave_qkv(QKV, cfg.n_heads, cfg.head_dim, 1)
        V = _deinterleave_qkv(QKV, cfg.n_heads, cfg.head_dim, 2)

    else:
        raise ValueError(f"Unknown arch '{arch}'. Supported: llama, neox, bloom.")

    return LayerWeights(Q, K, V, O, cfg)


def _deinterleave_qkv(
    QKV: torch.Tensor,
    n_heads: int,
    head_dim: int,
    which: int,   # 0=Q, 1=K, 2=V
) -> torch.Tensor:
    rows = []
    for h in range(n_heads):
        start = h * 3 * head_dim + which * head_dim
        rows.append(QKV[start: start + head_dim])
    return torch.cat(rows, dim=0)


def extract_mlp_weights(
    model_state_dict: dict,
    layer_idx: int,
    arch: str,
) -> dict:
    """Returns {"up": tensor_or_None, "down": tensor_or_None}."""
    sd = model_state_dict
    h = layer_idx
    f32 = dict(device="cpu", dtype=torch.float32)
    result = {"up": None, "down": None}

    if arch == "llama":
        if f"model.layers.{h}.mlp.up_proj.weight" in sd:
            result["up"] = sd[f"model.layers.{h}.mlp.up_proj.weight"].to(**f32)
        if f"model.layers.{h}.mlp.down_proj.weight" in sd:
            result["down"] = sd[f"model.layers.{h}.mlp.down_proj.weight"].to(**f32)
    elif arch == "bloom":
        if f"transformer.h.{h}.mlp.dense_h_to_4h.weight" in sd:
            result["up"] = sd[f"transformer.h.{h}.mlp.dense_h_to_4h.weight"].to(**f32)
        if f"transformer.h.{h}.mlp.dense_4h_to_h.weight" in sd:
            result["down"] = sd[f"transformer.h.{h}.mlp.dense_4h_to_h.weight"].to(**f32)

    return result


# ---------------------------------------------------------------------------
# Head similarity + Hungarian mapping
# ---------------------------------------------------------------------------

def compute_head_similarity_matrix(
    orig_weights: LayerWeights,
    new_weights: LayerWeights,
    Wh: Optional[torch.Tensor],
) -> np.ndarray:
    """
    Build (n_heads_orig, n_heads_new) cosine-similarity matrix.
    Each head pair is compared using flattened [W_QK, W_VO] vectors,
    with orig head projected into new model space via Wh.

    Vectorised: computes all head vectors in one pass then uses
    batched matrix multiply for pairwise cosine — ~100x faster on CPU.
    """
    cfg_o = orig_weights.cfg
    cfg_n = new_weights.cfg
    hd_o, hd_n = cfg_o.head_dim, cfg_n.head_dim

    logger.info(f"compute_head_similarity_matrix: "
                f"orig n_heads={cfg_o.n_heads} hd={hd_o} hidden={cfg_o.hidden_size} | "
                f"new  n_heads={cfg_n.n_heads} hd={hd_n} hidden={cfg_n.hidden_size} | "
                f"Wh={'None' if Wh is None else list(Wh.shape)}")

    # Build orig head vectors: project into new model space via Wh
    orig_vecs = []
    for a in range(cfg_o.n_heads):
        Q1 = orig_weights.Q[a * hd_o: (a + 1) * hd_o]
        K1 = orig_weights.K[a * hd_o: (a + 1) * hd_o]
        V1 = orig_weights.V[a * hd_o: (a + 1) * hd_o]
        O1 = orig_weights.O[:, a * hd_o: (a + 1) * hd_o]
        if Wh is not None:
            qk1 = Wh.T @ Q1.T @ K1 @ Wh
            vo1 = Wh.T @ V1.T @ O1.T @ Wh
        else:
            qk1 = Q1.T @ K1
            vo1 = V1.T @ O1.T
        orig_vecs.append(torch.cat([qk1.reshape(-1), vo1.reshape(-1)]))
    orig_mat = torch.stack(orig_vecs)   # (n_orig, vec_dim)

    # Build new head vectors (no Wh — already in new space)
    new_vecs = []
    for b in range(cfg_n.n_heads):
        Q2 = new_weights.Q[b * hd_n: (b + 1) * hd_n]
        K2 = new_weights.K[b * hd_n: (b + 1) * hd_n]
        V2 = new_weights.V[b * hd_n: (b + 1) * hd_n]
        O2 = new_weights.O[:, b * hd_n: (b + 1) * hd_n]
        qk2 = Q2.T @ K2
        vo2 = V2.T @ O2.T
        new_vecs.append(torch.cat([qk2.reshape(-1), vo2.reshape(-1)]))
    new_mat = torch.stack(new_vecs)   # (n_new, vec_dim)

    # Pairwise cosine similarity via normalised dot products
    orig_norm = orig_mat / (orig_mat.norm(dim=1, keepdim=True) + 1e-10)
    new_norm  = new_mat  / (new_mat.norm(dim=1, keepdim=True) + 1e-10)
    sim = (orig_norm @ new_norm.T).numpy().astype(np.float32)  # (n_orig, n_new)

    logger.info(f"  sim computed: min={sim.min():.4f}, max={sim.max():.4f}, mean={sim.mean():.4f}")
    return sim


def hungarian_head_mapping(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Returns head_mapping[orig_head] = new_head, maximising total similarity.
    Handles n_orig > n_new: transposes, assigns, then inverts.
    Unmapped source heads receive value -1.
    """
    n_orig, n_new = sim_matrix.shape
    logger.info(f"hungarian_head_mapping: ({n_orig}, {n_new})")
    if n_orig <= n_new:
        _, col_ind = linear_sum_assignment(-sim_matrix)
        return col_ind
    else:
        logger.info(f"  n_orig > n_new: {n_orig - n_new} heads will be unmapped")
        row_ind, col_ind = linear_sum_assignment(-sim_matrix.T)
        mapping = np.full(n_orig, -1, dtype=np.int64)
        for new_h, orig_h in zip(row_ind, col_ind):
            mapping[orig_h] = new_h
        logger.info(f"  unmapped source heads: {list(np.where(mapping == -1)[0])}")
        return mapping


# ---------------------------------------------------------------------------
# Equation (3): per-head LoRA transformation
# ---------------------------------------------------------------------------

def transform_attention_lora(
    orig_lora: dict,
    orig_weights: LayerWeights,
    new_weights: LayerWeights,
    head_mapping: np.ndarray,
    Wh: Optional[torch.Tensor],
    old_rank: int,
    orig_layer_idx: int,
    new_layer_idx: int,
    lora_key_fn_orig,
    lora_key_fn_new,
) -> dict:
    """
    Apply Equation (3) to transform Q/K/V/O LoRA weights, then SVD back.
    Works for llama-style (separate Q/K/V/O) architectures.
    For neox/bloom (fused QKV), see transform_fused_qkv_lora().
    """
    cfg_o = orig_weights.cfg
    cfg_n = new_weights.cfg
    hd_o, hd_n = cfg_o.head_dim, cfg_n.head_dim
    f32 = dict(device="cpu", dtype=torch.float32)

    # Load and expand original deltas
    delta_Q = _lora_delta(orig_lora, lora_key_fn_orig(orig_layer_idx, "q", "A"),
                          lora_key_fn_orig(orig_layer_idx, "q", "B"))
    delta_K = _lora_delta(orig_lora, lora_key_fn_orig(orig_layer_idx, "k", "A"),
                          lora_key_fn_orig(orig_layer_idx, "k", "B"))
    delta_V = _lora_delta(orig_lora, lora_key_fn_orig(orig_layer_idx, "v", "A"),
                          lora_key_fn_orig(orig_layer_idx, "v", "B"))
    delta_O = _lora_delta(orig_lora, lora_key_fn_orig(orig_layer_idx, "o", "A"),
                          lora_key_fn_orig(orig_layer_idx, "o", "B"))

    if cfg_o.n_rep > 1:
        delta_K = repeat_kv_heads(delta_K, hd_o, cfg_o.n_rep)
        delta_V = repeat_kv_heads(delta_V, hd_o, cfg_o.n_rep)

    # Allocate new deltas (expanded, will be collapsed later for GQA new model)
    new_delta_Q = torch.zeros(cfg_n.n_heads * hd_n, cfg_n.hidden_size, **f32)
    new_delta_K = torch.zeros(cfg_n.n_heads * hd_n, cfg_n.hidden_size, **f32)
    new_delta_V = torch.zeros(cfg_n.n_heads * hd_n, cfg_n.hidden_size, **f32)
    new_delta_O = torch.zeros(cfg_n.hidden_size, cfg_n.n_heads * hd_n, **f32)

    logger.info(f"transform_attention_lora: orig_layer={orig_layer_idx} → new_layer={new_layer_idx}")

    for a in range(cfg_o.n_heads):
        b = int(head_mapping[a])
        if b == -1:
            continue   # unmapped source head — target slot stays zero

        Q1 = orig_weights.Q[a * hd_o: (a + 1) * hd_o]
        K1 = orig_weights.K[a * hd_o: (a + 1) * hd_o]
        V1 = orig_weights.V[a * hd_o: (a + 1) * hd_o]
        O1 = orig_weights.O[:, a * hd_o: (a + 1) * hd_o]

        Q2 = new_weights.Q[b * hd_n: (b + 1) * hd_n]
        K2 = new_weights.K[b * hd_n: (b + 1) * hd_n]
        V2 = new_weights.V[b * hd_n: (b + 1) * hd_n]
        O2 = new_weights.O[:, b * hd_n: (b + 1) * hd_n]

        dQ1 = delta_Q[a * hd_o: (a + 1) * hd_o]
        dK1 = delta_K[a * hd_o: (a + 1) * hd_o]
        dV1 = delta_V[a * hd_o: (a + 1) * hd_o]
        dO1 = delta_O[:, a * hd_o: (a + 1) * hd_o]

        if Wh is not None:
            W_Q_x = Q1 @ Wh @ Q2.T
            new_delta_Q[b * hd_n: (b + 1) * hd_n] = W_Q_x.T @ dQ1 @ Wh

            W_K_x = K1 @ Wh @ K2.T
            new_delta_K[b * hd_n: (b + 1) * hd_n] = W_K_x.T @ dK1 @ Wh

            W_V_x = V1 @ Wh @ V2.T
            new_delta_V[b * hd_n: (b + 1) * hd_n] = W_V_x.T @ dV1 @ Wh

            W_O_x = O1.T @ Wh @ O2
            new_delta_O[:, b * hd_n: (b + 1) * hd_n] = Wh.T @ dO1 @ W_O_x
        else:
            W_Q_x = Q1 @ Q2.T
            new_delta_Q[b * hd_n: (b + 1) * hd_n] = W_Q_x.T @ dQ1

            W_K_x = K1 @ K2.T
            new_delta_K[b * hd_n: (b + 1) * hd_n] = W_K_x.T @ dK1

            W_V_x = V1 @ V2.T
            new_delta_V[b * hd_n: (b + 1) * hd_n] = W_V_x.T @ dV1

            W_O_x = O1.T @ O2
            new_delta_O[:, b * hd_n: (b + 1) * hd_n] = dO1 @ W_O_x

    # Collapse KV for GQA new model
    if cfg_n.n_rep > 1:
        new_delta_K = collapse_kv_heads(new_delta_K, cfg_n.n_kv_heads, cfg_n.n_rep, hd_n)
        new_delta_V = collapse_kv_heads(new_delta_V, cfg_n.n_kv_heads, cfg_n.n_rep, hd_n)

    result = {}
    for proj, delta_W in [("q", new_delta_Q), ("k", new_delta_K),
                           ("v", new_delta_V), ("o", new_delta_O)]:
        B, A = svd_low_rank(delta_W, old_rank)
        result[lora_key_fn_new(new_layer_idx, proj, "B")] = B
        result[lora_key_fn_new(new_layer_idx, proj, "A")] = A

    return result


def transform_fused_qkv_lora(
    orig_lora: dict,
    orig_weights: LayerWeights,
    new_weights: LayerWeights,
    head_mapping: np.ndarray,
    Wh: Optional[torch.Tensor],
    old_rank: int,
    orig_layer_idx: int,
    new_layer_idx: int,
    lora_key_fn_orig,
    lora_key_fn_new,
) -> dict:
    """
    Variant for architectures (neox, bloom) where Q/K/V are fused into a
    single QKV projection matrix, interleaved per head.

    delta_QKV layout: [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
    """
    cfg_o = orig_weights.cfg
    cfg_n = new_weights.cfg
    hd_o, hd_n = cfg_o.head_dim, cfg_n.head_dim
    f32 = dict(device="cpu", dtype=torch.float32)

    delta_QKV = _lora_delta(orig_lora,
                            lora_key_fn_orig(orig_layer_idx, "qkv", "A"),
                            lora_key_fn_orig(orig_layer_idx, "qkv", "B"))
    delta_O   = _lora_delta(orig_lora,
                            lora_key_fn_orig(orig_layer_idx, "o",   "A"),
                            lora_key_fn_orig(orig_layer_idx, "o",   "B"))

    new_delta_QKV = torch.zeros(cfg_n.n_heads * 3 * hd_n, cfg_n.hidden_size, **f32)
    new_delta_O   = torch.zeros(cfg_n.hidden_size, cfg_n.n_heads * hd_n, **f32)

    for a in range(cfg_o.n_heads):
        b = int(head_mapping[a])
        if b == -1:
            continue

        # Slice per-head Q, K, V from fused matrices
        Q1 = orig_weights.Q[a * hd_o: (a + 1) * hd_o]
        K1 = orig_weights.K[a * hd_o: (a + 1) * hd_o]
        V1 = orig_weights.V[a * hd_o: (a + 1) * hd_o]
        O1 = orig_weights.O[:, a * hd_o: (a + 1) * hd_o]

        Q2 = new_weights.Q[b * hd_n: (b + 1) * hd_n]
        K2 = new_weights.K[b * hd_n: (b + 1) * hd_n]
        V2 = new_weights.V[b * hd_n: (b + 1) * hd_n]
        O2 = new_weights.O[:, b * hd_n: (b + 1) * hd_n]

        # Slice per-head deltas from fused delta_QKV
        dQ1 = delta_QKV[a * 3 * hd_o:           a * 3 * hd_o + hd_o]
        dK1 = delta_QKV[a * 3 * hd_o + hd_o:    a * 3 * hd_o + 2 * hd_o]
        dV1 = delta_QKV[a * 3 * hd_o + 2 * hd_o:(a + 1) * 3 * hd_o]
        dO1 = delta_O[:, a * hd_o: (a + 1) * hd_o]

        if Wh is not None:
            W_Q_x = Q1 @ Wh @ Q2.T
            new_dQ = W_Q_x.T @ dQ1 @ Wh
            W_K_x = K1 @ Wh @ K2.T
            new_dK = W_K_x.T @ dK1 @ Wh
            W_V_x = V1 @ Wh @ V2.T
            new_dV = W_V_x.T @ dV1 @ Wh
            W_O_x = O1.T @ Wh @ O2
            new_delta_O[:, b * hd_n: (b + 1) * hd_n] = Wh.T @ dO1 @ W_O_x
        else:
            W_Q_x = Q1 @ Q2.T
            new_dQ = W_Q_x.T @ dQ1
            W_K_x = K1 @ K2.T
            new_dK = W_K_x.T @ dK1
            W_V_x = V1 @ V2.T
            new_dV = W_V_x.T @ dV1
            W_O_x = O1.T @ O2
            new_delta_O[:, b * hd_n: (b + 1) * hd_n] = dO1 @ W_O_x

        new_delta_QKV[b * 3 * hd_n:            b * 3 * hd_n + hd_n]    = new_dQ
        new_delta_QKV[b * 3 * hd_n + hd_n:     b * 3 * hd_n + 2 * hd_n] = new_dK
        new_delta_QKV[b * 3 * hd_n + 2 * hd_n: (b + 1) * 3 * hd_n]     = new_dV

    result = {}
    B, A = svd_low_rank(new_delta_QKV, old_rank)
    result[lora_key_fn_new(new_layer_idx, "qkv", "B")] = B
    result[lora_key_fn_new(new_layer_idx, "qkv", "A")] = A
    B, A = svd_low_rank(new_delta_O, old_rank)
    result[lora_key_fn_new(new_layer_idx, "o", "B")] = B
    result[lora_key_fn_new(new_layer_idx, "o", "A")] = A

    return result


# ---------------------------------------------------------------------------
# MLP LoRA transformation (direct A/B manipulation, no SVD needed)
# ---------------------------------------------------------------------------

def transform_mlp_lora(
    orig_lora: dict,
    orig_mlp: dict,   # {"up": tensor|None, "down": tensor|None}
    new_mlp: dict,
    Wh: Optional[torch.Tensor],
    orig_layer_idx: int,
    new_layer_idx: int,
    target_modules: list,
    lora_key_fn_orig,
    lora_key_fn_new,
) -> dict:
    """
    Transform MLP LoRA weights using direct A/B manipulation:

    up_proj   B_new = W_new_U @ pinv(Wh) @ pinv(W_old_U) @ B_old
              A_new = A_old @ Wh

    down_proj B_new = pinv(Wh) @ B_old
              A_new = A_old @ pinv(W_old_D) @ Wh @ W_new_D
    """
    result = {}
    f32 = dict(device="cpu", dtype=torch.float32)

    Wh_pinv = torch.linalg.pinv(Wh) if Wh is not None else None

    up_aliases   = ("up_proj", "dense_h_to_4h", "gate_proj")
    down_aliases = ("down_proj", "dense_4h_to_h")

    for mod in target_modules:
        is_up   = any(a in mod for a in up_aliases)
        is_down = any(a in mod for a in down_aliases)
        if not (is_up or is_down):
            continue

        key_B = lora_key_fn_orig(orig_layer_idx, mod, "B")
        key_A = lora_key_fn_orig(orig_layer_idx, mod, "A")
        if key_B not in orig_lora or key_A not in orig_lora:
            continue

        B_old = orig_lora[key_B].to(**f32)
        A_old = orig_lora[key_A].to(**f32)

        if is_up and orig_mlp.get("up") is not None and new_mlp.get("up") is not None:
            W_old_U = orig_mlp["up"]
            W_new_U = new_mlp["up"]
            if Wh is not None:
                B_new = W_new_U @ Wh_pinv @ torch.linalg.pinv(W_old_U) @ B_old
                A_new = A_old @ Wh
            else:
                B_new = W_new_U @ torch.linalg.pinv(W_old_U) @ B_old
                A_new = A_old

        elif is_down and orig_mlp.get("down") is not None and new_mlp.get("down") is not None:
            W_old_D = orig_mlp["down"]
            W_new_D = new_mlp["down"]
            if Wh is not None:
                B_new = Wh_pinv @ B_old
                A_new = A_old @ torch.linalg.pinv(W_old_D) @ Wh @ W_new_D
            else:
                B_new = B_old
                A_new = A_old @ torch.linalg.pinv(W_old_D) @ W_new_D

        else:
            logger.warning(f"Skipping MLP module '{mod}': missing weight matrices.")
            continue

        result[lora_key_fn_new(new_layer_idx, mod, "B")] = B_new
        result[lora_key_fn_new(new_layer_idx, mod, "A")] = A_new

    return result


# ---------------------------------------------------------------------------
# SVD low-rank decomposition
# ---------------------------------------------------------------------------

def svd_low_rank(delta_W: torch.Tensor, rank: int) -> tuple:
    """Decompose delta_W ≈ B @ A, returning (B, A) with rank singular values."""
    try:
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    except Exception as e:
        logger.error(f"SVD failed: {e}")
        d_out, d_in = delta_W.shape
        return torch.zeros(d_out, rank), torch.zeros(rank, d_in)
    r = min(rank, S.shape[0])
    scale = S[:r].sqrt()
    return U[:, :r] * scale, Vh[:r, :] * scale.unsqueeze(1)


# ---------------------------------------------------------------------------
# LoRA key construction
# ---------------------------------------------------------------------------

def make_lora_key_fn(arch: str):
    """
    Returns key_fn(layer_idx, proj, ab) -> str.

    proj for attention: "q", "k", "v", "o", "qkv" (fused neox/bloom)
    proj for MLP: the full module name, e.g. "up_proj", "down_proj"
    ab: "A" or "B"
    """
    _attn_modules = {
        "llama": {"q": "self_attn.q_proj", "k": "self_attn.k_proj",
                  "v": "self_attn.v_proj", "o": "self_attn.o_proj"},
        "neox":  {"qkv": "attention.query_key_value", "o": "attention.dense"},
        "bloom": {"qkv": "self_attention.query_key_value", "o": "self_attention.dense"},
    }
    _prefix = {
        "llama": "base_model.model.model.layers",
        "neox":  "base_model.model.gpt_neox.layers",
        "bloom": "base_model.model.transformer.h",
    }
    _mlp_prefix = {
        "llama": "mlp",
        "neox":  "mlp",
        "bloom": "mlp",
    }

    prefix = _prefix[arch]
    attn   = _attn_modules[arch]

    def key_fn(layer_idx: int, proj: str, ab: str) -> str:
        if proj in attn:
            module = attn[proj]
        else:
            module = f"{_mlp_prefix[arch]}.{proj}"
        return f"{prefix}.{layer_idx}.{module}.lora_{ab}.weight"

    return key_fn


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _lora_delta(lora: dict, key_A: str, key_B: str) -> torch.Tensor:
    A = lora[key_A].to(device="cpu", dtype=torch.float32)
    B = lora[key_B].to(device="cpu", dtype=torch.float32)
    return B @ A