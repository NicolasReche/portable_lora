import glob
import json
import os
import pandas as pd

# 1. Full experimental matrix definition
CONFIGS = [
    # ==================================
    # --- Target: LLaMA 3.2 3B ---
    # ==================================
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.2 3B (no module)",
        "file": "base_llama32_sentiment_seed7097.json",
        "Params": "—",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.2 3B (SFT)",
        "file": "identity_llama32_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.2 3B (RL V3)",
        "file": "rl_v3_llama32_to_llama32_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.2 3B (RL V4)",
        "file": "rl_v4_llama32_to_llama32_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    # --- Transfer from 3.1 to 3.2 (Zero-Shot) ---
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Zero-Shot SFT)",
        "file": "llama31_to_llama32_sentiment_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Zero-Shot RL V1)",
        "file": "rl_v1_llama31_to_llama32_sentiment_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Zero-Shot RL V2)",
        "file": "rl_v2_llama31_to_llama32_sentiment_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Zero-Shot RL V2.1)",
        "file": "rl_v2_1_llama31_to_llama32_sentiment_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Zero-Shot RL V3)",
        "file": "rl_v3_llama31_to_llama32_sentiment_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Zero-Shot RL V4)",
        "file": "rl_v4_llama31_to_llama32_sentiment_seed7097.json",
        "Params": "0.4x",
    },
    # --- Transfer from 3.1 to 3.2 (Few-Step Adaptation) ---
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Few-Step SFT)",
        "file": "few_step_llama31_to_llama32_sentiment_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Few-Step RL V1)",
        "file": "few_step_rl_llama31_to_llama32_sentiment_v1_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Few-Step RL V2)",
        "file": "few_step_rl_llama31_to_llama32_sentiment_v2_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Few-Step RL V2.1)",
        "file": "few_step_rl_llama31_to_llama32_sentiment_v2_1_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Few-Step RL V3)",
        "file": "few_step_rl_llama31_to_llama32_sentiment_v3_seed7097.json",
        "Params": "0.4x",
    },
    {
        "Target": "LLaMA 3.2 3B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.2 3B (Few-Step RL V4)",
        "file": "few_step_rl_llama31_to_llama32_sentiment_v4_seed7097.json",
        "Params": "0.4x",
    },

    # ==================================
    # --- Target: LLaMA 3.1 8B ---
    # ==================================
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.1 8B (no module)",
        "file": "base_llama31_sentiment_seed7097.json",
        "Params": "—",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.1 8B (SFT)",
        "file": "llama31_to_llama31_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.1 8B (RL V1)",
        "file": "rl_v1_llama31_to_llama31_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.1 8B (RL V2)",
        "file": "rl_v2_llama31_to_llama31_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.1 8B (RL V2.1)",
        "file": "rl_v2_1_llama31_to_llama31_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.1 8B (RL V3)",
        "file": "rl_v3_llama31_to_llama31_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.1 8B → LLaMA 3.1 8B (RL V4)",
        "file": "rl_v4_llama31_to_llama31_sentiment_seed7097.json",
        "Params": "1.0x",
    },
    # --- Transfer from 3.2 to 3.1 (Zero-Shot) ---
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Zero-Shot SFT)",
        "file": "llama32_to_llama31_sentiment_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Zero-Shot RL V1)",
        "file": "rl_v1_llama32_to_llama31_sentiment_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Zero-Shot RL V2)",
        "file": "rl_v2_llama32_to_llama31_sentiment_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Zero-Shot RL V2.1)",
        "file": "rl_v2_1_llama32_to_llama31_sentiment_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Zero-Shot RL V3)",
        "file": "rl_v3_llama32_to_llama31_sentiment_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Zero-Shot RL V4)",
        "file": "rl_v4_llama32_to_llama31_sentiment_seed7097.json",
        "Params": "2.7x",
    },
    # --- Transfer from 3.2 to 3.1 (Few-Step Adaptation) ---
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Few-Step SFT)",
        "file": "few_step_llama32_to_llama31_sentiment_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Few-Step RL V1)",
        "file": "few_step_rl_llama32_to_llama31_sentiment_v1_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Few-Step RL V2)",
        "file": "few_step_rl_llama32_to_llama31_sentiment_v2_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Few-Step RL V2.1)",
        "file": "few_step_rl_llama32_to_llama31_sentiment_v2_1_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Few-Step RL V3)",
        "file": "few_step_rl_llama32_to_llama31_sentiment_v3_seed7097.json",
        "Params": "2.7x",
    },
    {
        "Target": "LLaMA 3.1 8B",
        "Setting": "LLaMA 3.2 3B → LLaMA 3.1 8B (Few-Step RL V4)",
        "file": "few_step_rl_llama32_to_llama31_sentiment_v4_seed7097.json",
        "Params": "2.7x",
    },
]


def parse_pipeline_results():
    log_files = sorted(
        glob.glob("logs/*analyze*.out"), key=os.path.getmtime, reverse=True
    )
    if not log_files:
        print("No analysis log files found in logs/!")
        return {}

    latest_log = log_files[0]
    results = {}
    current_model = None
    current_set = None

    with open(latest_log, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("--- ") and line.endswith(" ---"):
                raw_path = line.replace("---", "").strip()
                current_model = os.path.basename(raw_path)
                results[current_model] = {}
            elif line.startswith("Test Set:"):
                current_set = line.replace("Test Set:", "").strip()
                if current_model:
                    results[current_model][current_set] = {}
            elif line.startswith("- accuracy:"):
                acc = float(line.split(":")[1].strip())
                if current_model and current_set:
                    results[current_model][current_set]["accuracy"] = acc
            elif line.startswith("- distinct-n:"):
                dist = float(line.split(":")[1].strip())
                if current_model and current_set:
                    results[current_model][current_set]["distinct-n"] = dist
            elif line.startswith("- slor:"):
                slor = float(line.split(":")[1].strip())
                if current_model and current_set:
                    results[current_model][current_set]["slor"] = slor

    return results


parsed_data = parse_pipeline_results()
rows = []

for cfg in CONFIGS:
    fname = cfg["file"]
    m_data = parsed_data.get(fname, {})

    pplm_acc = (
        m_data.get("data/pplm_prompts.csv", {}).get("accuracy", 0.0) * 100
    )
    sts_acc = (
        m_data.get("data/sts_benchmark_test_subset.csv", {}).get("accuracy", 0.0) * 100
    )
    sts_proc_acc = (
        m_data.get("data/sts_benchmark_processed.csv", {}).get("accuracy", 0.0) * 100
    )

    acc_list = [v for v in [pplm_acc, sts_acc, sts_proc_acc] if v > 0]
    avg_all = sum(acc_list) / len(acc_list) if acc_list else 0.0

    slors = [
        m_data.get(k, {}).get("slor", 0.0)
        for k in m_data
        if "slor" in m_data.get(k, {})
    ]
    dists = [
        m_data.get(k, {}).get("distinct-n", 0.0)
        for k in m_data
        if "distinct-n" in m_data.get(k, {})
    ]

    avg_slor = sum(slors) / len(slors) if slors else 0.0
    avg_dist = sum(dists) / len(dists) if dists else 0.0

    if avg_all > 0 or fname in ["base_llama32_sentiment_seed7097.json", "base_llama31_sentiment_seed7097.json"]:
        rows.append({
            "Target": cfg["Target"],
            "Setting": cfg["Setting"],
            "Avg all": round(avg_all, 2),
            "PPLM": round(pplm_acc, 2),
            "STS": round(sts_acc, 2),
            "STS proc": round(sts_proc_acc, 2),
            "Dist-n": round(avg_dist, 3),
            "SLOR": round(avg_slor, 2),
            "Params": cfg["Params"],
        })

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_excel("results/sentiment_control_benchmark_dynamic.xlsx", index=False)
df.to_csv("results/sentiment_control_benchmark_dynamic.csv", index=False)
print("Updated results/sentiment_control_benchmark_dynamic.xlsx successfully.")
print(df.to_string(index=False))
