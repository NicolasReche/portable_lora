"""
analysis/make_tables.py

Reads the JSON results from run_experiments.py and produces:
  - Table 2-equivalent: Sentiment control (CE, Diversity, Fluency)
  - Table 3-equivalent: Topic control
  - Table 4-equivalent: Multi-attribute control
  - NEW Table 5: Portability results (CE_original vs CE_ported_0step)

Output: CSV files + printed LaTeX-ready tables.

Usage:
    python analysis/make_tables.py --results_dir ./results
"""

import os
import json
import argparse
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals=2) -> str:
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Table 2/3: Single-attribute control
# Columns: CE_avg, CE_yelp/agnews, CE_imdb/dbpedia, CE_sst2, dist1, dist2, dist3, SLOR
# ---------------------------------------------------------------------------

def make_single_attribute_table(
    results: dict,
    attribute: str,
    composition_modes: list[str],
) -> list[dict]:
    """
    Build result rows for a single attribute.

    Result keys expected (from phase2_composition):
        single_{attribute}                — single adapter eval
        composed_{mode}_{attribute}       — cross-attribute composition eval
    """
    rows = []

    # Raw model baseline placeholder
    rows.append({"technique": "Raw model", "CE": None,
                 "dist1": None, "dist2": None, "dist3": None, "SLOR": None})

    # Single adapter
    r = results.get(f"single_{attribute}", {})
    ds_label = "Yelp" if attribute == "sentiment" else "AG News"
    rows.append({
        "technique": f"+ QLoRA RL {ds_label}",
        "CE":    r.get("CE"),
        "dist1": r.get("dist1"),
        "dist2": r.get("dist2"),
        "dist3": r.get("dist3"),
        "SLOR":  r.get("SLOR"),
    })

    # Cross-attribute compositions
    mode_label = {
        "sum":            "Output Summing",
        "average":        "Output Averaging",
        "weight_average": "Averaged Weights",
    }
    for mode in composition_modes:
        r = results.get(f"composed_{mode}_{attribute}", {})
        rows.append({
            "technique": f"+ QLoRA RL {mode_label.get(mode, mode)} (S+T)",
            "CE":    r.get("CE"),
            "dist1": r.get("dist1"),
            "dist2": r.get("dist2"),
            "dist3": r.get("dist3"),
            "SLOR":  r.get("SLOR"),
        })

    return rows


def print_single_attribute_table(rows: list[dict], title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = f"{'Technique':<55} {'CE':>7} {'dist1':>7} {'dist2':>7} {'dist3':>7} {'SLOR':>7}"
    print(header)
    print("-" * 80)
    for r in rows:
        ce    = fmt(r.get("CE"), 1)
        d1    = fmt(r.get("dist1"), 3)
        d2    = fmt(r.get("dist2"), 3)
        d3    = fmt(r.get("dist3"), 3)
        slor  = fmt(r.get("SLOR"), 2)
        print(f"{r['technique']:<55} {ce:>7} {d1:>7} {d2:>7} {d3:>7} {slor:>7}")


def save_csv(rows: list[dict], path: str):
    if not rows:
        return
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# NEW Table 5: Portability results
# ---------------------------------------------------------------------------

def make_portability_table(results: dict) -> list[dict]:
    """
    Columns: attribute, dataset, CE_original, CE_ported_0step, CE_delta, CE_retention
    """
    rows = []
    for key, r in results.items():
        if not key.startswith("port_"):
            continue
        # key format: port_{attribute}_{dataset}_{target_model}
        parts = key.split("_")
        attribute = parts[1]
        # dataset and target_model are variable-length — just use the full key
        rows.append({
            "experiment":       key,
            "attribute":        attribute,
            "CE_original":      r.get("CE_original"),
            "CE_ported_0step":  r.get("CE_ported_0step"),
            "CE_delta":         r.get("CE_delta"),
            "CE_retention_%":   r.get("CE_retention", 0) * 100
                                if r.get("CE_retention") is not None else None,
        })
    return rows


def print_portability_table(rows: list[dict]):
    print(f"\n{'='*80}")
    print(f"  Portability Results (0 fine-tuning steps)")
    print(f"{'='*80}")
    header = (f"{'Experiment':<45} {'CE_orig':>8} {'CE_port':>8} "
              f"{'Δ CE':>8} {'Retention%':>11}")
    print(header)
    print("-" * 80)
    for r in rows:
        print(
            f"{r['experiment']:<45} "
            f"{fmt(r['CE_original'], 1):>8} "
            f"{fmt(r['CE_ported_0step'], 1):>8} "
            f"{fmt(r['CE_delta'], 1):>8} "
            f"{fmt(r['CE_retention_%'], 1):>11}"
        )


# ---------------------------------------------------------------------------
# SFT vs RL comparison summary
# ---------------------------------------------------------------------------

def print_comparison_summary(
    rl_results: dict,
    sft_baseline: dict | None = None,
):
    """
    Print a summary comparing RL vs SFT CE scores.
    sft_baseline: dict mapping the same keys to CE values from your previous paper.
    """
    print(f"\n{'='*80}")
    print("  RL vs SFT Comparison Summary")
    print(f"{'='*80}")

    if sft_baseline is None:
        print("  (No SFT baseline provided — showing RL results only)")
        for k, v in rl_results.items():
            if isinstance(v, dict) and "CE" in v:
                print(f"  {k:<50} CE={fmt(v['CE'], 1)}")
        return

    print(f"{'Experiment':<50} {'CE_SFT':>8} {'CE_RL':>8} {'RL-SFT':>8}")
    print("-" * 70)
    for k in sorted(rl_results.keys()):
        rl_val = rl_results[k].get("CE") if isinstance(rl_results[k], dict) else None
        sft_val = sft_baseline.get(k)
        if rl_val is None:
            continue
        delta = (rl_val - sft_val) if sft_val is not None else None
        print(
            f"{k:<50} {fmt(sft_val, 1):>8} {fmt(rl_val, 1):>8} "
            f"{fmt(delta, 1):>8}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir",  type=str, default="./tables")
    parser.add_argument(
        "--sft_baseline", type=str, default=None,
        help="Path to JSON file with SFT CE scores for comparison"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all result files
    comp_results = load_json(os.path.join(args.results_dir, "phase2_composition.json"))
    port_results = load_json(os.path.join(args.results_dir, "phase3_portability.json"))
    port_comp_results = load_json(os.path.join(args.results_dir, "phase4_composed_ported.json"))
    sft_baseline = load_json(args.sft_baseline) if args.sft_baseline else None

    composition_modes = ["sum", "average", "weight_average"]

    # ---- Table 2: Sentiment control ----
    sent_rows = make_single_attribute_table(
        comp_results,
        attribute="sentiment",
        composition_modes=composition_modes,
    )
    print_single_attribute_table(sent_rows, "Sentiment Control (RL-trained, Yelp)")
    save_csv(sent_rows, os.path.join(args.output_dir, "table_sentiment.csv"))

    # ---- Table 3: Topic control ----
    topic_rows = make_single_attribute_table(
        comp_results,
        attribute="topic",
        composition_modes=composition_modes,
    )
    print_single_attribute_table(topic_rows, "Topic Control (RL-trained, AG News)")
    save_csv(topic_rows, os.path.join(args.output_dir, "table_topic.csv"))

    # ---- Table 5 (new): Portability ----
    port_rows = make_portability_table(port_results)
    if port_rows:
        print_portability_table(port_rows)
        save_csv(port_rows, os.path.join(args.output_dir, "table_portability.csv"))
    else:
        print("\n  No portability results found (Phase 3 not run yet).")

    # ---- Table 6 (new): Composed ported modules ----
    if port_comp_results:
        comp_port_rows = []
        for k, v in port_comp_results.items():
            comp_port_rows.append({
                "experiment":    k,
                "CE_sentiment":  v.get("CE_sentiment"),
                "CE_topic":      v.get("CE_topic"),
                "dist1":         v.get("dist1"),
            })
        print(f"\n{'='*80}")
        print("  Composed Ported Modules")
        print(f"{'='*80}")
        print(f"{'Experiment':<50} {'CE_sent':>8} {'CE_topic':>8} {'dist1':>7}")
        print("-" * 75)
        for r in comp_port_rows:
            print(
                f"{r['experiment']:<50} "
                f"{fmt(r['CE_sentiment'], 1):>8} "
                f"{fmt(r['CE_topic'], 1):>8} "
                f"{fmt(r['dist1'], 3):>7}"
            )
        save_csv(comp_port_rows,
                 os.path.join(args.output_dir, "table_composed_ported.csv"))

    # ---- RL vs SFT comparison ----
    print_comparison_summary(comp_results, sft_baseline)

    print(f"\nAll tables saved to {args.output_dir}/")


if __name__ == "__main__":
    main()