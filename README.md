# Modular Controlled Text Generation: Portability of Trained LoRA Modules

## Overview & Motivation

### The Problem

Controlled Text Generation (CTG) typically requires fine-tuning an entire Large Language Model (LLM) or training heavy, architecture-specific steering components. While Parameter-Efficient Fine-Tuning (PEFT) methods like Low-Rank Adaptation (LoRA) have made adapting models highly efficient, these trained modules are traditionally locked to the exact base model architecture they were trained on. If a new, more efficient model family or iteration is released, researchers and developers are forced to expend computational resources to retrain their control modules from scratch.

### The Approach

This research investigates a highly practical question: **Can a specialized LoRA module trained on one base LLM be directly "plugged in" and ported to an entirely different model architecture or family?**

By decoupling the control module from its native base model, we evaluate the cross-model zero-shot capability of attribute-specific adapters (e.g., controlling `sentiment` or `topic`). We benchmark these boundaries across distinct open-weights model ecosystem iterations, including:

- Generational variations (e.g., porting between `LLaMA 3.1` and `LLaMA 3.2`)
- Cross-family architectures (e.g., porting from `LLaMA 3.1` to `Qwen 3`)

When Zero-Shot portability fails, we introduce **Few-Step Adaptation** and **Reinforcement Learning (RL)** to bridge the architectural gap and salvage the module's behavior without full retraining.

## Repository Structure

```text
├── config/          # YAML configuration files for training and evaluation
├── jobs/            # Slurm cluster bash job scripts (.job)
├── pipeline/        # High-level bash scripts orchestrating the full execution pipelines
├── outputs/         # Generated text, predictions, and evaluation outputs
└── scripts/         # Python execution scripts for training, inference, and evaluation
```

## Environment Setup

The repository uses `uv` for fast, reproducible Python virtual environment and dependency management (or standard `pip` inside `.venv`).

1. **Activate the Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

2. **Verify Environment**:
    You can use the provided script to verify your CUDA and package installations:
    ```bash
    bash pipeline/check_env.sh
    ```

## Execution & Pipeline

The pipeline consists of multiple sequential stages executed via Slurm workload manager jobs. We have orchestrated these into easy-to-use bash scripts located in the `pipeline/` directory.

### 1. Baseline Reproduction (SFT & Zero-Shot Portability)

Trains base attribute-specific LoRA modules using Supervised Fine-Tuning (SFT) and evaluates their direct Zero-Shot portability on target models. 

- **Execution**: `bash pipeline/remaining_tasks/run_repro_topic.sh`
- **What it does**: 
  - SFT Training of the control module.
  - Identity Inference (Source model + Source module).
  - Zero-Shot Inference (Target model + Source module).
  - Few-Step Adaptation (Quick retraining on target).

### 2. Reinforcement Learning (RL) Pipeline

To overcome the limitations of SFT and Zero-Shot transfer, we apply Reinforcement Learning (PPO) to directly optimize the modules for attribute control, diversity, and fluency. The RL algorithm underwent several iterations to fix learning flaws:

- **V1**: Raw control reward. Causes diversity loss.
- **V2**: Stronger control penalty. Causes "Mode Collapse".
- **V3.2 & V3.3**: Introduces `Distinct-n` penalty and empirical SLOR (Wikitext unigram probabilities) to perfectly balance control, diversity, and natural fluency.
- **V4.1 & V4.2**: Replaces `Distinct-n` with 3/4-gram Shannon Entropy to maintain diversity without penalizing natural language distributions (Zipf's law).

**Execution**:
- **Sentiment RL**: `bash pipeline/remaining_tasks/run_rl_sentiment.sh`
- **Topic RL**: `bash pipeline/remaining_tasks/run_rl_topic.sh`

### 3. Evaluation & Analysis

Calculates metrics determining generation quality and attribute control capabilities across the portability boundaries. 
Metrics evaluated include:
- **PPLM Accuracy** (Control accuracy)
- **SLOR** (Fluency and natural language likelihood)
- **Distinct-n / Entropy** (Lexical diversity)

- **Execution**: `bash pipeline/analyse.sh`
- **Results**: The aggregated insights and historical evolution of the algorithms are consolidated in `results/results_analysis.md`.

## Configurations Details

- **Training Configs** (`config/sft_train_*.yaml`): Set up specific LoRA rank hyper-parameters (*r=32, α=64*), datasets configurations (`Yelp` for sentiment, `AG News` for topics), and optimizer setups (`paged_adamw_32bit` using `bf16`).
- **Evaluation Configs** (`config/evaluate_*.yaml`): Specify metric tracking frameworks along with target classification boundaries (`Positive`/`Negative` or `Sports`/`Science/Technology`/`Business`/`World`).
