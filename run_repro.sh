#!/bin/bash
# run_repro.sh - Baseline Reproduction Pipeline (SFT, Porting, Adaptation)

echo "Submitting Baseline Reproduction Pipeline (Sentiment)..."
mkdir -p logs outputs predictions results models

# 1. Train SFT LoRA adapters (Llama 3.1 8B & Llama 3.2 3B)
echo "[1/9] Training SFT control modules..."
JOB_SFT=$(sbatch --parsable jobs/sft_train.job)

# 2. Raw base model inference & evaluation
echo "[2/9] Base model inference..."
JOB_INF_BASE=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/inference_base_models.job)

echo "[3/9] Base model evaluation..."
JOB_EVAL_BASE=$(sbatch --parsable --dependency=afterok:$JOB_INF_BASE jobs/evaluate_base_models.job)

# 3. Identity inference & evaluation (Module on its native model)
echo "[4/9] Identity inference..."
JOB_INF_ID=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/inference_identity_sentiment.job)

echo "[5/9] Identity evaluation..."
JOB_EVAL_ID=$(sbatch --parsable --dependency=afterok:$JOB_INF_ID jobs/evaluate_identity.job)

# 4. Zero-shot adapter porting & evaluation (8B <-> 3B)
echo "[6/9] Zero-shot ported module inference..."
JOB_INF_PORT=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/inference_llama31_module_sentiment.job)

echo "[7/9] Zero-shot ported module evaluation..."
JOB_EVAL_PORT=$(sbatch --parsable --dependency=afterok:$JOB_INF_PORT jobs/evaluate_llama31_module.job)

# 5. Few-step adaptation & evaluation (8B <-> 3B)
echo "[8/9] Few-step adaptation..."
JOB_FEW_STEP=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/train_post_porting_sft.job)

echo "[9/9] Few-step evaluation..."
JOB_EVAL_FEW_STEP=$(sbatch --parsable --dependency=afterok:$JOB_FEW_STEP jobs/evaluate_few_step.job)

echo "=========================================================="
echo "Reproduction jobs submitted successfully."
echo "Track status with: squeue -u $USER"
echo "=========================================================="
