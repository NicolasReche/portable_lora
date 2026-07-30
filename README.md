# Modular Controlled Text Generation: Portability of Trained LoRA Modules

## Overview & Motivation

### The Problem

Controlled Text Generation (CTG) typically requires fine-tuning an entire Large Language Model (LLM) or training heavy, architecture-specific steering components. While Parameter-Efficient Fine-Tuning (PEFT) methods like Low-Rank Adaptation (LoRA) have made adapting models highly efficient, these trained modules are traditionally locked to the exact base model architecture they were trained on. If a new, more efficient model family or iteration is released, researchers and developers are forced to expend computational resources to retrain their control modules from scratch.


### The Approach

This research investigates a highly practical question: **Can a specialized LoRA module trained on one base LLM be directly "plugged in" and ported to an entirely different model architecture or family?**

By decoupling the control module from its native base model, we evaluate the cross-model zero-shot capability of attribute-specific adapters (e.g., controlling `sentiment` or `topic`). We benchmark these boundaries across distinct open-weights model ecosystem iterations, including:

- Generational variations (e.g., porting between `LLaMA 3.1` and `LLaMA 3.2`)
- Cross-family architectures (e.g., porting from `LLaMA 3.1` to `Qwen 3`)

The ultimate goal of this section is to determine the architectural and semantic boundaries of LoRA portability, paving the way for truly modular, hot-swappable text control without retraining overhead.

## Repository Structure

```text
├── config/          # YAML configuration files for training and evaluation
├── jobs/            # Slurm cluster bash job scripts (.sh)
├── outputs/         # Generated text, predictions, and evaluation outputs
└── scripts/         # Python execution scripts for training, inference, and evaluation
```

## Environment Setup

The repository uses `uv` for fast, reproducible Python virtual environment and dependency management.

1. **Activate the Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

2. **Install Dependencies**:
    Ensure all dependencies matching the configuration frameworks (Hugging Face ecosystem, PEFT, PyTorch) are synchronized using `uv`.

## Execution & Pipeline

The pipeline consists of three sequential stages executed via Slurm workload manager jobs: **Training (SFT)**, **Portability Inference**, and **Evaluation**.

### 1. Module Training (Supervised Fine-Tuning)

Trains attribute-specific LoRA modules (e.g., `sentiment`, `topic`) using base configurations across designated models and seeds.

- Job File: `jobs/sft_train.job`
- Resource Required: 1x NVIDIA A100 GPU
- Command:
```bash
    # Iterates through attributes (sentiment, topic), seeds, and model types
    uv run scripts/train_sft_model.py \
        --config_path config/sft_train_${model}_${attribute}.yaml \
        --seed $seed \
        --model_dir models \
        --run_name sft_${model}_${attribute}_seed${seed}
```

### 2. Portability Inference (Cross-Model Generation)

Tests the direct portability of a LoRA module trained on a source model (e.g., `LLaMA 3.1 8B`) by plugging it into alternative target architectures (`LLaMA 3.2 3B`, `Qwen 3 4B`, `Qwen 3 8B`) to generate controlled text.

- Job File: `jobs/inference_llama31_module_topic.job`
- Resource Required: 1x NVIDIA A100 GPU (90GB Memory allocation)
- Command:
```bash
    uv run scripts/run_portability.py \
        --step generate \
        --seed $seed \
        --module models/sft_llama3.1_8b_topic_seed${seed}/checkpoint-1840/ \
        --src_model llama31 \
        --tgt_model $model \
        --attribute topic \
        --dataset data/pplm_prompts.csv data/sts_benchmark_processed.csv data/sts_benchmark_test_subset.csv \
        --output outputs/topic/llama31_to_${model}_topic_seed${seed}.json
```

### 3. Evaluation Pipeline

Calculates metrics determining generation quality and attribute control capabilities across the portability boundaries. Metrics evaluated include **SLOR**, **Accuracy**, and **Distinct-n**.

- Job File: `evaluate_llama31_module.job`
- Resource Required: 1x NVIDIA RTX A6000 GPU
-Command:
```bash
    uv run scripts/main_eval.py \
        --config evaluate_${attribute}.yaml \
        --seed $seed \
        --output_file outputs/llama31_to_${model}_${attribute}_seed${seed}.json \
        --predictions_file predictions/llama31_to_${model}_${attribute}_seed${seed}.json
```

## Configurations Details

- **Training Configs** (`config/sft_train_*.yaml`): Set up specific LoRA rank hyper-parameters (*r=32, α=64*, target modules mapping all linear layers), datasets configurations (`Yelp` for sentiment, `AG News` for topics), and optimizer setups (`paged_adamw_32bit` using `bf16`).

- **Evaluation Configs** (`config/evaluate_*.yaml`): Specify metric tracking frameworks along with target classification boundaries (`Positive`/`Negative` or `Sports`/`Science`/`Technology`/`Business`/`World`).
