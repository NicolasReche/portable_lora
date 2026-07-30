"""
analysis/make_portability_table.py

Reads the per-seed JSON result files produced by run_portability.py and
aggregates them into the portability table shown in the thesis.

File naming convention:
    outputs/{src}_to_{tgt}_{attribute}_seed{seed}.json

Seeds: 7097, 4613, 89723

Aggregation rules:
  - CE, Distinct-n, and SLOR are weighted averages across test sets,
    using the number of generated texts as the weight for each test set.
  - Then the weighted average is taken across the 3 seeds (equal weight,
    since each seed operates on the same dataset).
  - "Avg port." (the first column) is the simple mean of CE across all
    test sets and both control classes (overall CE).
  - Distinct-n [d1, d2, d3] is taken from evaluation_results['distinct-n']['overall']
    within each test set, then averaged across seeds.
  - SLOR is taken from evaluation_results['slor']['overall'] within each
    test set, then averaged across seeds.
  - Params ratio = n_tgt_params / n_src_params (reported in the JSON if
    present, otherwise filled from a hardcoded lookup table).

Usage:
    python analysis/make_portability_table.py \\
        --results_dir ./outputs \\
        --attribute sentiment \\
        --output_csv ./tables/portability_sentiment.csv \\
        --latex

The script prints a human-readable table to stdout, writes a CSV, and
optionally writes a LaTeX table.
"""

import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [7097, 4613, 89723]

# Canonical test-set short names, in the order they appear in the table.
# Keys must match the substring of the file path stored in the JSON.
TEST_SET_LABELS = {
    "pplm":     "PPLM",
    "sts_benchmark_processed": "STS proc",
    "sts_benchmark_test":      "STS",
}

# Transfer condition for each (src, tgt) pair — determines the row group.
TRANSFER_CONDITIONS = {
    # Identity
    ("llama32", "llama32"): "identity",
    ("llama31", "llama31"): "identity",
    ("qwen34b", "qwen34b"): "identity",
    ("qwen38b", "qwen38b"): "identity",
    # Same family, different size
    ("llama32", "llama31"): "same_family_diff_size",
    ("llama31", "llama32"): "same_family_diff_size",
    ("qwen34b", "qwen38b"): "same_family_diff_size",
    ("qwen38b", "qwen34b"): "same_family_diff_size",
    # Cross-family, comparable size
    ("llama32", "qwen34b"): "cross_family_comparable",
    ("llama31", "qwen38b"): "cross_family_comparable",
    ("qwen34b", "llama32"): "cross_family_comparable",
    ("qwen38b", "llama31"): "cross_family_comparable",
    # Cross-family, different size
    ("llama32", "qwen38b"): "cross_family_diff_size",
    ("llama31", "qwen34b"): "cross_family_diff_size",
    ("qwen34b", "llama31"): "cross_family_diff_size",
    ("qwen38b", "llama32"): "cross_family_diff_size",
}

CONDITION_ORDER = [
    "base_model",
    "identity",
    "same_family_diff_size",
    "cross_family_comparable",
    "cross_family_diff_size",
]

CONDITION_LABELS = {
    "base_model":              "RAW BASE MODEL (no module)",
    "identity":                "IDENTITY",
    "same_family_diff_size":   "SAME FAMILY, DIFFERENT SIZE",
    "cross_family_comparable": "CROSS-FAMILY, COMPARABLE SIZE",
    "cross_family_diff_size":  "CROSS-FAMILY, DIFFERENT SIZE",
}

# Canonical order of base models in the base-model section
BASE_MODEL_ORDER = ["llama32", "llama31", "qwen34b", "qwen38b"]

# Approx parameter ratios: n_tgt_params / n_src_params
# Used when the JSON does not contain this information.
PARAMS_RATIO = {
    # (src, tgt)
    ("llama32", "llama32"): 1.0,
    ("llama31", "llama31"): 1.0,
    ("qwen34b", "qwen34b"): 1.0,
    ("qwen38b", "qwen38b"): 1.0,
    ("llama32", "llama31"): 2.7,   # 8B / 3B
    ("llama31", "llama32"): 0.4,   # 3B / 8B
    ("qwen34b", "qwen38b"): 2.0,   # 8B / 4B
    ("qwen38b", "qwen34b"): 0.5,   # 4B / 8B
    ("llama32", "qwen34b"): 1.3,   # ~4B / 3B
    ("llama31", "qwen38b"): 1.0,   # 8B / 8B
    ("qwen34b", "llama32"): 0.75,  # 3B / 4B
    ("qwen38b", "llama31"): 1.0,   # 8B / 8B
    ("llama32", "qwen38b"): 2.7,   # 8B / 3B
    ("llama31", "qwen34b"): 0.5,   # 4B / 8B
    ("qwen34b", "llama31"): 2.0,   # 8B / 4B
    ("qwen38b", "llama32"): 0.4,   # 3B / 8B
}

# Canonical row ordering within each condition group (src -> tgt)
PAIR_ORDER = [
    ("llama32", "llama32"),
    ("llama31", "llama31"),
    ("qwen34b", "qwen34b"),
    ("qwen38b", "qwen38b"),
    ("llama32", "llama31"),
    ("llama31", "llama32"),
    ("qwen34b", "qwen38b"),
    ("qwen38b", "qwen34b"),
    ("llama32", "qwen34b"),
    ("llama31", "qwen38b"),
    ("qwen34b", "llama32"),
    ("qwen38b", "llama31"),
    ("llama32", "qwen38b"),
    ("llama31", "qwen34b"),
    ("qwen34b", "llama31"),
    ("qwen38b", "llama32"),
]

MODEL_LABELS = {
    "llama32": "LLaMA 3.2 3B",
    "llama31": "LLaMA 3.1 8B",
    "qwen34b": "Qwen3 4B",
    "qwen38b": "Qwen3 8B",
}

# Canonical target-model order for the by-target layout
TGT_ORDER = ["llama32", "llama31", "qwen34b", "qwen38b"]

# Short condition labels used as sub-headings in the by-target layout
CONDITION_LABELS_SHORT = {
    "base_model":              "Raw base (no module)",
    "identity":                "Identity",
    "same_family_diff_size":   "Same family, diff. size",
    "cross_family_comparable": "Cross-family, comp. size",
    "cross_family_diff_size":  "Cross-family, diff. size",
}


# ---------------------------------------------------------------------------
# JSON loading and parsing
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_result_files(results_dir: str, attribute: str) -> dict:
    """
    Scan results_dir for files matching the naming pattern and return a dict:
        {(src, tgt, seed): filepath}
    """
    pattern = re.compile(
        r"(?P<src>[^_/]+)_to_(?P<tgt>[^_]+)_" + re.escape(attribute) +
        r"_seed(?P<seed>\d+)\.json$"
    )
    found = {}
    for p in Path(results_dir).glob("*.json"):
        m = pattern.match(p.name)
        if m:
            src  = m.group("src")
            tgt  = m.group("tgt")
            seed = int(m.group("seed"))
            found[(src, tgt, seed)] = str(p)
    return found


def _test_set_key(path_str: str) -> Optional[str]:
    """
    Map a test-set file path (as stored in the JSON) to its short label key.
    Returns None if unrecognised.
    """
    for key in TEST_SET_LABELS:
        if key in path_str:
            return key
    return None


def _n_samples(test_set_data: dict) -> int:
    """Number of generated texts in this test set — used as the weight."""
    return len(test_set_data.get("generated_texts", []))


def find_base_files(results_dir: str, attribute: str) -> dict:
    """
    Scan results_dir for base-model result files and return:
        {(model, seed): filepath}

    Expected naming: base_{model}_{attribute}_seed{seed}.json
    """
    pattern = re.compile(
        r"base_(?P<model>[^_]+)_" + re.escape(attribute) +
        r"_seed(?P<seed>\d+)\.json$"
    )
    found = {}
    for p in Path(results_dir).glob("*.json"):
        m = pattern.match(p.name)
        if m:
            model = m.group("model")
            seed  = int(m.group("seed"))
            found[(model, seed)] = str(p)
    return found


def build_base_rows(results_dir: str, attribute: str) -> list[dict]:
    """
    Load base-model result files and return one aggregated row per model,
    in BASE_MODEL_ORDER.  These rows use condition='base_model', src=model,
    tgt=None, and params_ratio=None (no module attached).
    """
    file_map = find_base_files(results_dir, attribute)

    # Group by model
    model_data: dict[str, list[dict]] = {}
    for (model, seed), fpath in sorted(file_map.items()):
        if model not in model_data:
            model_data[model] = []
        try:
            data = load_json(fpath)
            metrics = parse_single_result(data)
            metrics["seed"] = seed
            model_data[model].append(metrics)
        except Exception as e:
            print(f"  WARNING: Could not parse {fpath}: {e}")

    rows = []
    for model in BASE_MODEL_ORDER:
        seed_metrics = model_data.get(model, [])
        agg = aggregate_seeds(seed_metrics) if seed_metrics else _empty_metrics()
        if not seed_metrics:
            agg["n_seeds"] = 0

        rows.append({
            "src":           model,
            "tgt":           None,
            "src_label":     MODEL_LABELS.get(model, model),
            "tgt_label":     None,
            "condition":     "base_model",
            "n_seeds":       agg.get("n_seeds", 0),
            "ce_avg":        agg.get("ce_avg"),
            "ce_pplm":       agg.get("ce_pplm"),
            "ce_sts":        agg.get("ce_sts"),
            "ce_sts_proc":   agg.get("ce_sts_proc"),
            "dist1":         agg.get("dist1"),
            "dist2":         agg.get("dist2"),
            "dist3":         agg.get("dist3"),
            "slor":          agg.get("slor"),
            "params_ratio":  None,   # no module, ratio not applicable
            "ce_avg_std":    agg.get("ce_avg_std"),
            "dist1_std":     agg.get("dist1_std"),
            "slor_std":      agg.get("slor_std"),
        })

    return rows


def parse_single_result(data: dict) -> dict:
    """
    Parse one JSON result file into a flat metrics dict.

    Returns:
        {
          "ce_pplm":      float,   # CE overall for PPLM test set
          "ce_sts":       float,
          "ce_sts_proc":  float,
          "ce_avg":       float,   # weighted average across test sets
          "dist1":        float,   # weighted avg across test sets
          "dist2":        float,
          "dist3":        float,
          "slor":         float,
          "n_total":      int,     # total samples across all test sets
        }
    Returns all None values if data is missing.
    """
    test_sets = data.get("test_sets", {})
    if not test_sets:
        return _empty_metrics()

    # Accumulate weighted sums
    ce_by_label   = {}   # short_label -> ce
    dist_weighted  = [0.0, 0.0, 0.0]
    slor_weighted  = 0.0
    ce_weighted    = 0.0
    total_n        = 0

    for ts_path, ts_data in test_sets.items():
        label_key = _test_set_key(ts_path)
        n = _n_samples(ts_data)
        if n == 0:
            continue

        er = ts_data.get("evaluation_results", {})
        if not er:
            continue

        # CE
        acc = er.get("accuracy", {})
        ce_overall = acc.get("overall")
        if ce_overall is None:
            continue

        ce_weighted += ce_overall * n
        total_n     += n

        if label_key:
            ce_by_label[label_key] = ce_overall

        # Distinct-n: collect all class_* keys (works for both sentiment
        # ["class_negative","class_positive"] and topic
        # ["class_business","class_science/technology","class_sports","class_world"]).
        dn = er.get("distinct-n", {})
        class_lists = [v for k, v in dn.items()
                       if k.startswith("class_") and isinstance(v, list)]
        if class_lists:
            for i in range(min(3, min(len(lst) for lst in class_lists))):
                avg_i = sum(lst[i] for lst in class_lists) / len(class_lists)
                dist_weighted[i] += avg_i * n
        else:
            # Fallback: integer keys "1","2","3"
            int_vals = [dn.get(str(k)) for k in (1, 2, 3)]
            if all(v is not None for v in int_vals):
                for i, v in enumerate(int_vals):
                    dist_weighted[i] += v * n
            else:
                # Last resort: "overall" as a list [d1, d2, d3]
                overall_list = dn.get("overall")
                if isinstance(overall_list, list) and len(overall_list) >= 3:
                    for i in range(3):
                        dist_weighted[i] += overall_list[i] * n

        # SLOR
        slor = er.get("slor", {})
        slor_overall = slor.get("overall")
        if slor_overall is not None:
            slor_weighted += slor_overall * n

    if total_n == 0:
        return _empty_metrics()

    return {
        "ce_pplm":     ce_by_label.get("pplm"),
        "ce_sts":      ce_by_label.get("sts_benchmark_test"),
        "ce_sts_proc": ce_by_label.get("sts_benchmark_processed"),
        "ce_avg":      ce_weighted / total_n,
        "dist1":       dist_weighted[0] / total_n,
        "dist2":       dist_weighted[1] / total_n,
        "dist3":       dist_weighted[2] / total_n,
        "slor":        slor_weighted / total_n,
        "n_total":     total_n,
    }


def _empty_metrics() -> dict:
    return {
        "ce_pplm": None, "ce_sts": None, "ce_sts_proc": None,
        "ce_avg": None, "dist1": None, "dist2": None, "dist3": None,
        "slor": None, "n_total": 0,
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def aggregate_seeds(seed_metrics: list[dict]) -> dict:
    """
    Average metrics across seeds.  Uses equal weights across seeds
    (all seeds operate on the same dataset, so equal weighting is correct).
    Missing values (None) are excluded from the average.
    """
    scalar_keys = ["ce_pplm", "ce_sts", "ce_sts_proc", "ce_avg",
                   "dist1", "dist2", "dist3", "slor"]
    result = {}
    for k in scalar_keys:
        vals = [m[k] for m in seed_metrics if m.get(k) is not None]
        result[k] = sum(vals) / len(vals) if vals else None
        result[f"{k}_std"] = _std(vals) if len(vals) > 1 else None
    result["n_seeds"] = len(seed_metrics)
    return result


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def build_table(results_dir: str, attribute: str) -> list[dict]:
    """
    Load all result files and return a list of row dicts ordered as:
      1. Base model rows (no module attached)
      2. Portability rows grouped by transfer condition
    """
    # --- Base model rows ---
    base_rows = build_base_rows(results_dir, attribute)

    # --- Portability rows ---
    file_map = find_result_files(results_dir, attribute)
    if not file_map:
        print(f"WARNING: No portability result files found in '{results_dir}' "
              f"for attribute='{attribute}'.")
        print("  Expected pattern: {{src}}_to_{{tgt}}_{attribute}_seed{{seed}}.json")

    pair_data: dict[tuple, list[dict]] = {}
    for (src, tgt, seed), fpath in sorted(file_map.items()):
        key = (src, tgt)
        if key not in pair_data:
            pair_data[key] = []
        try:
            data = load_json(fpath)
            metrics = parse_single_result(data)
            metrics["seed"] = seed
            pair_data[key].append(metrics)
        except Exception as e:
            print(f"  WARNING: Could not parse {fpath}: {e}")

    port_rows = []
    for src, tgt in PAIR_ORDER:
        if (src, tgt) not in TRANSFER_CONDITIONS:
            continue
        condition = TRANSFER_CONDITIONS[(src, tgt)]
        seed_metrics = pair_data.get((src, tgt), [])

        if seed_metrics:
            agg = aggregate_seeds(seed_metrics)
        else:
            agg = _empty_metrics()
            agg["n_seeds"] = 0

        params_ratio = PARAMS_RATIO.get((src, tgt), None)

        port_rows.append({
            "src":           src,
            "tgt":           tgt,
            "src_label":     MODEL_LABELS.get(src, src),
            "tgt_label":     MODEL_LABELS.get(tgt, tgt),
            "condition":     condition,
            "n_seeds":       agg.get("n_seeds", 0),
            "ce_avg":        agg.get("ce_avg"),
            "ce_pplm":       agg.get("ce_pplm"),
            "ce_sts":        agg.get("ce_sts"),
            "ce_sts_proc":   agg.get("ce_sts_proc"),
            "dist1":         agg.get("dist1"),
            "dist2":         agg.get("dist2"),
            "dist3":         agg.get("dist3"),
            "slor":          agg.get("slor"),
            "params_ratio":  params_ratio,
            "ce_avg_std":    agg.get("ce_avg_std"),
            "dist1_std":     agg.get("dist1_std"),
            "slor_std":      agg.get("slor_std"),
        })

    return base_rows + port_rows


# ---------------------------------------------------------------------------
# LFT file discovery
# ---------------------------------------------------------------------------

def find_lft_files(results_dir: str, attribute: str) -> dict:
    """
    Scan results_dir for LFT result files and return:
        {(src, tgt, seed): filepath}

    Expected naming: lft_{src}_to_{tgt}_{attribute}_seed{seed}.json
    """
    pattern = re.compile(
        r"lft_(?P<src>[^_]+)_to_(?P<tgt>[^_]+)_" + re.escape(attribute) +
        r"_seed(?P<seed>\d+)\.json$"
    )
    found = {}
    for p in Path(results_dir).glob("*.json"):
        m = pattern.match(p.name)
        if m:
            src  = m.group("src")
            tgt  = m.group("tgt")
            seed = int(m.group("seed"))
            found[(src, tgt, seed)] = str(p)
    return found


def _load_port_pair_data(results_dir: str, attribute: str,
                          file_finder) -> dict:
    """
    Generic helper: scan files with `file_finder`, parse and aggregate by
    (src, tgt) pair.  Returns {(src, tgt): aggregated_metrics_dict}.
    """
    file_map = file_finder(results_dir, attribute)
    pair_data: dict[tuple, list[dict]] = {}
    for (src, tgt, seed), fpath in sorted(file_map.items()):
        key = (src, tgt)
        if key not in pair_data:
            pair_data[key] = []
        try:
            data = load_json(fpath)
            metrics = parse_single_result(data)
            metrics["seed"] = seed
            pair_data[key].append(metrics)
        except Exception as e:
            print(f"  WARNING: Could not parse {fpath}: {e}")
    return pair_data


def _make_port_row(src: str, tgt: str, pair_data: dict,
                   file_type: str = "zero_shot") -> dict:
    """
    Build a single portability row dict for (src, tgt) from aggregated data.
    file_type is stored for downstream filtering ('zero_shot' or 'lft').
    """
    condition = TRANSFER_CONDITIONS.get((src, tgt), "unknown")
    seed_metrics = pair_data.get((src, tgt), [])
    agg = aggregate_seeds(seed_metrics) if seed_metrics else _empty_metrics()
    if not seed_metrics:
        agg["n_seeds"] = 0

    return {
        "src":           src,
        "tgt":           tgt,
        "src_label":     MODEL_LABELS.get(src, src),
        "tgt_label":     MODEL_LABELS.get(tgt, tgt),
        "condition":     condition,
        "file_type":     file_type,
        "n_seeds":       agg.get("n_seeds", 0),
        "ce_avg":        agg.get("ce_avg"),
        "ce_pplm":       agg.get("ce_pplm"),
        "ce_sts":        agg.get("ce_sts"),
        "ce_sts_proc":   agg.get("ce_sts_proc"),
        "dist1":         agg.get("dist1"),
        "dist2":         agg.get("dist2"),
        "dist3":         agg.get("dist3"),
        "slor":          agg.get("slor"),
        "params_ratio":  PARAMS_RATIO.get((src, tgt), None),
        "ce_avg_std":    agg.get("ce_avg_std"),
        "dist1_std":     agg.get("dist1_std"),
        "slor_std":      agg.get("slor_std"),
    }


# ---------------------------------------------------------------------------
# By-target table layout
# ---------------------------------------------------------------------------

def build_table_by_target(results_dir: str, attribute: str) -> list[dict]:
    """
    Build rows grouped by TARGET model.

    Within each target-model block the order is:
      1. Raw base model (that target, no module)
      2. Identity       (src == tgt)
      3. Same-family, different size
      4. Cross-family, comparable size
      5. Cross-family, different size

    This makes it easy to compare all sources ported into the same backbone.
    """
    base_by_model = {r["src"]: r
                     for r in build_base_rows(results_dir, attribute)}
    pair_data = _load_port_pair_data(results_dir, attribute,
                                     find_result_files)

    rows = []
    for tgt in TGT_ORDER:
        # 1. Base model row for this target
        if tgt in base_by_model:
            base_row = dict(base_by_model[tgt])
            base_row["tgt_group"] = tgt   # tag for grouping in LaTeX
            rows.append(base_row)

        # 2. Collect all (src, tgt) rows, ordered by condition priority
        tgt_pairs = [(src, t) for (src, t) in PAIR_ORDER if t == tgt]
        condition_priority = {
            "identity":                0,
            "same_family_diff_size":   1,
            "cross_family_comparable": 2,
            "cross_family_diff_size":  3,
        }
        tgt_pairs.sort(key=lambda st: condition_priority.get(
            TRANSFER_CONDITIONS.get(st, ""), 99))

        for src, t in tgt_pairs:
            row = _make_port_row(src, t, pair_data, file_type="zero_shot")
            row["tgt_group"] = tgt
            rows.append(row)

    return rows


def build_lft_table_by_target(results_dir: str, attribute: str) -> list[dict]:
    """
    Like build_table_by_target, but substitutes LFT results for zero-shot
    porting rows.  Identity rows keep their zero-shot results (no porting
    distortion was introduced, so LFT does not apply).  Base model rows
    are unchanged.

    Within each target-model block:
      1. Raw base model    (zero-shot, as reference)
      2. Identity          (zero-shot, as upper bound)
      3. Same-family       -> LFT result
      4. Cross-family      -> LFT result
    """
    base_by_model = {r["src"]: r
                     for r in build_base_rows(results_dir, attribute)}
    zs_pair_data  = _load_port_pair_data(results_dir, attribute,
                                         find_result_files)
    lft_pair_data = _load_port_pair_data(results_dir, attribute,
                                         find_lft_files)

    if not lft_pair_data:
        print(f"WARNING: No LFT result files found in '{results_dir}' "
              f"for attribute='{attribute}'.")
        print("  Expected pattern: lft_{{src}}_to_{{tgt}}_{attribute}_seed{{seed}}.json")

    rows = []
    for tgt in TGT_ORDER:
        # 1. Base model row
        if tgt in base_by_model:
            base_row = dict(base_by_model[tgt])
            base_row["tgt_group"] = tgt
            rows.append(base_row)

        # 2+. All (src, tgt) pairs, ordered by condition
        tgt_pairs = [(src, t) for (src, t) in PAIR_ORDER if t == tgt]
        condition_priority = {
            "identity":                0,
            "same_family_diff_size":   1,
            "cross_family_comparable": 2,
            "cross_family_diff_size":  3,
        }
        tgt_pairs.sort(key=lambda st: condition_priority.get(
            TRANSFER_CONDITIONS.get(st, ""), 99))

        for src, t in tgt_pairs:
            condition = TRANSFER_CONDITIONS.get((src, t), "unknown")
            if condition == "identity":
                # Identity: no porting, use zero-shot result as upper bound
                row = _make_port_row(src, t, zs_pair_data, file_type="zero_shot")
            else:
                # Non-identity: use LFT result
                row = _make_port_row(src, t, lft_pair_data, file_type="lft")
            row["tgt_group"] = tgt
            rows.append(row)

    return rows




def fmt(val, decimals: int = 2, pct: bool = False, dash: str = "—") -> str:
    if val is None:
        return dash
    if pct:
        return f"{val * 100:.{decimals}f}"
    return f"{val:.{decimals}f}"


def fmt_ratio(val) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}$\\times$"


# ---------------------------------------------------------------------------
# Plain-text table printer
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], attribute: str,
                by_target: bool = False, mode: str = "zero-shot"):
    col_w = {
        "pair":       32,
        "ce_avg":      8,
        "ce_pplm":     8,
        "ce_sts":      8,
        "ce_stsp":     9,
        "d1":          7,
        "d2":          7,
        "d3":          7,
        "slor":        8,
        "ratio":       8,
    }
    sep = "-" * 107
    header = (
        f"{'Setting':<{col_w['pair']}}"
        f"{'CE avg':>{col_w['ce_avg']}}"
        f"{'PPLM':>{col_w['ce_pplm']}}"
        f"{'STS':>{col_w['ce_sts']}}"
        f"{'STS proc':>{col_w['ce_stsp']}}"
        f"{'dist-1':>{col_w['d1']}}"
        f"{'dist-2':>{col_w['d2']}}"
        f"{'dist-3':>{col_w['d3']}}"
        f"{'SLOR':>{col_w['slor']}}"
        f"{'Params':>{col_w['ratio']}}"
    )

    title_mode = f"{mode.upper()} results"
    print(f"\n{'='*107}")
    print(f"  Portability {title_mode} — {attribute.upper()}")
    print(f"{'='*107}")

    if by_target:
        current_tgt = None
        for r in rows:
            tgt_group = r.get("tgt_group", r.get("tgt") or r.get("src"))
            if tgt_group != current_tgt:
                current_tgt = tgt_group
                tgt_label = MODEL_LABELS.get(tgt_group, tgt_group)
                print(f"\n  TARGET: {tgt_label}")
                print(f"  {sep}")
                print(f"  {header}")
                print(f"  {sep}")

            if r["condition"] == "base_model":
                pair_str = f"{r['src_label']} (no module)"
            elif r["condition"] == "identity":
                pair_str = f"{r['src_label']} → {r['tgt_label']} (identity)"
            else:
                pair_str = f"{r['src_label']} → {r['tgt_label']}"

            seeds_ok = r["n_seeds"]
            line = (
                f"  {pair_str:<{col_w['pair']}}"
                f"{fmt(r['ce_avg'],  2, pct=True):>{col_w['ce_avg']}}"
                f"{fmt(r['ce_pplm'], 2, pct=True):>{col_w['ce_pplm']}}"
                f"{fmt(r['ce_sts'],  2, pct=True):>{col_w['ce_sts']}}"
                f"{fmt(r['ce_sts_proc'], 2, pct=True):>{col_w['ce_stsp']}}"
                f"{fmt(r['dist1'], 3):>{col_w['d1']}}"
                f"{fmt(r['dist2'], 3):>{col_w['d2']}}"
                f"{fmt(r['dist3'], 3):>{col_w['d3']}}"
                f"{fmt(r['slor'],  2):>{col_w['slor']}}"
                f"{fmt_ratio(r['params_ratio']):>{col_w['ratio']}}"
            )
            missing = f"  [{seeds_ok}/3 seeds]" if seeds_ok < 3 else ""
            print(line + missing)

    else:
        current_cond = None
        for r in rows:
            if r["condition"] != current_cond:
                current_cond = r["condition"]
                print(f"\n  {CONDITION_LABELS[current_cond]}")
                print(f"  {sep}")
                print(f"  {header}")
                print(f"  {sep}")

            pair_str = (
                r["src_label"]
                if r["condition"] == "base_model"
                else f"{r['src_label']} → {r['tgt_label']}"
            )
            seeds_ok = r["n_seeds"]
            line = (
                f"  {pair_str:<{col_w['pair']}}"
                f"{fmt(r['ce_avg'],  2, pct=True):>{col_w['ce_avg']}}"
                f"{fmt(r['ce_pplm'], 2, pct=True):>{col_w['ce_pplm']}}"
                f"{fmt(r['ce_sts'],  2, pct=True):>{col_w['ce_sts']}}"
                f"{fmt(r['ce_sts_proc'], 2, pct=True):>{col_w['ce_stsp']}}"
                f"{fmt(r['dist1'], 3):>{col_w['d1']}}"
                f"{fmt(r['dist2'], 3):>{col_w['d2']}}"
                f"{fmt(r['dist3'], 3):>{col_w['d3']}}"
                f"{fmt(r['slor'],  2):>{col_w['slor']}}"
                f"{fmt_ratio(r['params_ratio']):>{col_w['ratio']}}"
            )
            missing = f"  [{seeds_ok}/3 seeds]" if seeds_ok < 3 else ""
            print(line + missing)

    print(f"\n  CE values are percentages. "
          f"Distinct-n and SLOR are weighted averages across test sets and seeds.")
    print(f"{'='*107}\n")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    fieldnames = [
        "src", "tgt", "src_label", "tgt_label", "condition", "n_seeds",
        "ce_avg", "ce_pplm", "ce_sts", "ce_sts_proc",
        "dist1", "dist2", "dist3", "slor", "params_ratio",
        "ce_avg_std", "dist1_std", "slor_std",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV: {path}")


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

METRIC_COLS = ["ce_avg", "ce_pplm", "ce_sts", "ce_sts_proc",
               "dist1", "dist2", "dist3", "slor"]


def rank_within_groups(rows: list[dict]) -> list[dict]:
    """
    For each target-model group, rank every numeric metric column and annotate
    each row with sets 'rank1_cols' and 'rank2_cols' containing the column
    names where that row holds the best or second-best value.

    Higher is always better for all metrics (CE, Distinct-n, SLOR).
    Rows with None values are excluded from ranking for that column.
    Ties: both rows sharing the top value both get rank-1; the next
    distinct value gets rank-2.
    """
    from itertools import groupby

    rows = [dict(r) for r in rows]   # shallow copy so we don't mutate originals
    for r in rows:
        r["rank1_cols"] = set()
        r["rank2_cols"] = set()

    # Group by tgt_group
    groups: dict[str, list] = {}
    for r in rows:
        key = r.get("tgt_group", r.get("tgt") or r.get("src"))
        groups.setdefault(key, []).append(r)

    for tgt, group in groups.items():
        for col in METRIC_COLS:
            # Collect (value, row) pairs where value is not None
            scored = [(r[col], r) for r in group if r.get(col) is not None]
            if not scored:
                continue

            # Sort descending — all metrics are higher-is-better
            scored.sort(key=lambda x: x[0], reverse=True)

            best_val = scored[0][0]
            # Mark all rows tied at best value as rank-1
            for val, r in scored:
                if val == best_val:
                    r["rank1_cols"].add(col)
                else:
                    break

            # Find second-best: first distinct value below best
            second_val = None
            for val, r in scored:
                if val < best_val:
                    second_val = val
                    break
            if second_val is not None:
                for val, r in scored:
                    if val == second_val:
                        r["rank2_cols"].add(col)

    return rows


def save_latex(rows: list[dict], path: str, attribute: str,
               caption: str = "", label: str = "",
               by_target: bool = False,
               show_params: bool = True):
    r"""
    Produce a LaTeX table.

    When by_target=False (default): groups rows by transfer condition,
    one \midrule between each group.

    When by_target=True: groups rows by target model (tgt_group key),
    with \midrule between target blocks and an italic sub-heading showing
    the target model name.  Within each block the condition is printed as
    a short label in the Setting column instead of a separate heading row.
    """
    os.makedirs(Path(path).parent, exist_ok=True)

    if not caption:
        if by_target:
            caption = (
                r"Portability results grouped by target model backbone. "
                r"Each block shows the raw base model (no module), the identity "
                r"upper bound, and all source models ported into that target. "
                r"All values are averages over 3 seeds. "
                r"CE values are percentages; Distinct-$n$ and SLOR are weighted "
                r"averages across test sets and seeds."
            )
        else:
            caption = (
                r"Portability results for SFT-trained LoRA modules ported across "
                r"model backbones with 0 post-porting learning steps. "
                r"All values are averages over 3 seeds. "
                r"CE values are percentages; Distinct-$n$ and SLOR are weighted "
                r"averages across test sets and seeds."
            )
    if not label:
        suffix = "by_target" if by_target else "by_condition"
        label = f"tab:portability_{attribute}_{suffix}"

    def lx(val, decimals=2, pct=False, col=None, r=None):
        if val is None:
            return "---"
        if pct:
            formatted = f"{val * 100:.{decimals}f}"
        else:
            formatted = f"{val:.{decimals}f}"
        if col and r:
            if col in r.get("rank1_cols", set()):
                return r"\textbf{" + formatted + r"}"
            if col in r.get("rank2_cols", set()):
                return r"\textit{" + formatted + r"}"
        return formatted

    col_spec = r"lrrrrrrrrrr" if show_params else r"lrrrrrrrrr"
    params_header = r" & \textbf{Params}" if show_params else ""

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"\multicolumn{2}{c}{} & "
        r"\multicolumn{4}{c}{\textbf{Control Effectiveness} $\uparrow$} & "
        r"\multicolumn{3}{c}{\textbf{Distinct-$n$} $\uparrow$} & & \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
        r"\textbf{Setting} & \textbf{Avg all} & \textbf{PPLM} & "
        r"\textbf{STS} & \textbf{STS proc} & "
        r"$d_1$ & $d_2$ & $d_3$ & \textbf{SLOR} $\uparrow$"
        + params_header + r" \\",
        r"\midrule",
    ]

    if by_target:
        # Annotate rows with rank1_cols / rank2_cols before rendering
        rows = rank_within_groups(rows)

        current_tgt = None
        for r in rows:
            tgt_group = r.get("tgt_group", r.get("tgt") or r.get("src"))

            if tgt_group != current_tgt:
                if current_tgt is not None:
                    lines.append(r"\midrule")
                current_tgt = tgt_group
                tgt_label = MODEL_LABELS.get(tgt_group, tgt_group)
                lines.append(
                    r"\multicolumn{10}{l}{\textit{Target: "
                    + tgt_label + r"}} \\"
                )

            if r["condition"] == "base_model":
                setting = r["src_label"] + r" \textit{(no module)}"
                params_str = "---"
            elif r["condition"] == "identity":
                setting = (r["src_label"] + r" $\rightarrow$ " + r["tgt_label"]
                           + r" \textit{(identity)}")
                params_str = fmt_ratio(r["params_ratio"])
            else:
                setting = (r["src_label"] + r" $\rightarrow$ " + r["tgt_label"])
                params_str = fmt_ratio(r["params_ratio"])

            row_str = (
                f"{setting} & "
                f"{lx(r['ce_avg'],    1, pct=True, col='ce_avg',    r=r)} & "
                f"{lx(r['ce_pplm'],   1, pct=True, col='ce_pplm',   r=r)} & "
                f"{lx(r['ce_sts'],    1, pct=True, col='ce_sts',    r=r)} & "
                f"{lx(r['ce_sts_proc'],1,pct=True, col='ce_sts_proc',r=r)} & "
                f"{lx(r['dist1'], 3, col='dist1', r=r)} & "
                f"{lx(r['dist2'], 3, col='dist2', r=r)} & "
                f"{lx(r['dist3'], 3, col='dist3', r=r)} & "
                f"{lx(r['slor'],  2, col='slor',  r=r)}"
                + (f" & {params_str}" if show_params else "")
                + r" \\"
            )
            lines.append(row_str)

    else:
        # Original layout: group by condition (no ranking)
        current_cond = None
        for r in rows:
            if r["condition"] != current_cond:
                if current_cond is not None:
                    lines.append(r"\midrule")
                current_cond = r["condition"]
                label_str = CONDITION_LABELS[current_cond]
                lines.append(
                    r"\multicolumn{10}{l}{\textit{" + label_str.title() + r"}} \\"
                )

            if r["condition"] == "base_model":
                pair = r["src_label"]
                params_str = "---"
            else:
                pair = r["src_label"] + r" $\rightarrow$ " + r["tgt_label"]
                params_str = fmt_ratio(r["params_ratio"])

            row_str = (
                f"{pair} & "
                f"{lx(r['ce_avg'], 1, pct=True)} & "
                f"{lx(r['ce_pplm'], 1, pct=True)} & "
                f"{lx(r['ce_sts'], 1, pct=True)} & "
                f"{lx(r['ce_sts_proc'], 1, pct=True)} & "
                f"{lx(r['dist1'], 3)} & "
                f"{lx(r['dist2'], 3)} & "
                f"{lx(r['dist3'], 3)} & "
                f"{lx(r['slor'], 2)}"
                + (f" & {params_str}" if show_params else "")
                + r" \\"
            )
            lines.append(row_str)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved LaTeX: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate portability results and produce the thesis table."
    )
    parser.add_argument(
        "--results_dir", type=str, default="./outputs",
        help="Directory containing the per-seed JSON result files.",
    )
    parser.add_argument(
        "--attribute", type=str, default="sentiment",
        choices=["sentiment", "topic"],
        help="Which attribute's results to aggregate.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tables",
        help="Directory to write CSV and LaTeX outputs.",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Also write a LaTeX table.",
    )
    parser.add_argument(
        "--by-target", action="store_true", dest="by_target",
        help="Group rows by target model instead of transfer condition.",
    )
    parser.add_argument(
        "--lft", action="store_true",
        help="Show LFT results instead of zero-shot porting results "
             "(identity and base rows are always zero-shot). "
             "Implies --by-target.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="Which seeds to aggregate (default: 7097 4613 89723).",
    )
    args = parser.parse_args()

    # --lft always uses the by-target layout
    if args.lft:
        args.by_target = True

    if args.seeds != SEEDS:
        import make_portability_table as _self
        _self.SEEDS = args.seeds

    print(f"\nAggregating portability results from: {args.results_dir}")
    print(f"Attribute: {args.attribute}  |  Seeds: {SEEDS}"
          f"  |  Layout: {'by-target' if args.by_target else 'by-condition'}"
          f"  |  Mode: {'LFT' if args.lft else 'zero-shot'}")

    # --- Build rows ---
    if args.lft:
        rows = build_lft_table_by_target(args.results_dir, args.attribute)
        table_tag = "lft"
    elif args.by_target:
        rows = build_table_by_target(args.results_dir, args.attribute)
        table_tag = "zero_shot_by_target"
    else:
        rows = build_table(args.results_dir, args.attribute)
        table_tag = "zero_shot_by_condition"

    # --- Print ---
    print_table(rows, args.attribute,
                by_target=args.by_target,
                mode="LFT" if args.lft else "zero-shot")

    # --- Save CSV ---
    csv_path = os.path.join(
        args.output_dir, f"portability_{args.attribute}_{table_tag}.csv"
    )
    save_csv(rows, csv_path)

    # --- Save LaTeX ---
    if args.latex:
        tex_path = os.path.join(
            args.output_dir, f"portability_{args.attribute}_{table_tag}.tex"
        )
        save_latex(rows, tex_path, args.attribute,
                   by_target=args.by_target)
        if args.lft:
            # Override caption/label for LFT table, and drop Params column
            lft_caption = (
                r"Post-porting light fine-tuning (LFT) results grouped by target "
                r"model backbone. Identity rows show zero-shot results (upper bound); "
                r"all other porting rows show results after LFT on the target backbone. "
                r"Base model rows show uncontrolled generation (reference). "
                r"All values are averages over 3 seeds. "
                r"CE values are percentages; Distinct-$n$ and SLOR are weighted "
                r"averages across test sets and seeds."
            )
            save_latex(rows, tex_path, args.attribute,
                       caption=lft_caption,
                       label=f"tab:portability_{args.attribute}_lft",
                       by_target=True,
                       show_params=False)

    # --- Coverage summary ---
    port_rows = [r for r in rows if r["condition"] not in ("base_model",)]
    base_rows  = [r for r in rows if r["condition"] == "base_model"]
    identity_rows = [r for r in port_rows if r["condition"] == "identity"]
    porting_rows  = [r for r in port_rows if r["condition"] != "identity"]
    mode_label = "LFT" if args.lft else "Zero-shot port"

    def _coverage(subset):
        n_total    = len(subset)
        n_complete = sum(1 for r in subset if r["n_seeds"] >= len(SEEDS))
        n_partial  = sum(1 for r in subset if 0 < r["n_seeds"] < len(SEEDS))
        n_missing  = sum(1 for r in subset if r["n_seeds"] == 0)
        return n_total, n_complete, n_partial, n_missing

    print(f"\n  {'Type':<22} {'Total':>5} {'Complete':>9} {'Partial':>8} {'Missing':>8}")
    print(f"  {'-'*55}")
    for label, subset in [
        ("Base model",     base_rows),
        ("Identity",       identity_rows),
        (f"{mode_label}", porting_rows),
    ]:
        tot, comp, part, miss = _coverage(subset)
        print(f"  {label:<22} {tot:>5} {comp:>9} {part:>8} {miss:>8}")

    # Detail on partial and missing porting rows
    partial = [r for r in porting_rows if 0 < r["n_seeds"] < len(SEEDS)]
    missing = [r for r in porting_rows if r["n_seeds"] == 0]

    if partial:
        print(f"\n  Partial results ({[len(SEEDS)]} seeds expected):")
        for r in partial:
            print(f"    [{r['n_seeds']}/{len(SEEDS)}]  "
                  f"{r['src_label']} -> {r['tgt_label']}")
    if missing:
        print(f"\n  No results yet:")
        for r in missing:
            print(f"    [ 0/{len(SEEDS)}]  {r['src_label']} -> {r['tgt_label']}")


if __name__ == "__main__":
    main()