#!/bin/bash
# run_repro.sh - Baseline Reproduction Pipeline (SFT, Porting, Adaptation)

echo "Submitting Baseline Reproduction Pipeline (Sentiment)..."
mkdir -p logs outputs predictions results models

SEED=7097

check_and_run() {
    local step_name=$1
    local check_file=$2
    local job_script=$3
    local dependency=$4

    if [ -f "$check_file" ] || [ -d "$check_file" ]; then
        echo "  [SKIP] $step_name (Found: $check_file)" >&2
        echo ""
    else
        local sbatch_cmd="sbatch --parsable"
        # Only add dependency if valid numeric Job ID is passed
        if [[ -n "$dependency" && "$dependency" =~ ^[0-9]+$ ]]; then
            sbatch_cmd="$sbatch_cmd --dependency=afterok:$dependency"
        fi
        
        job_id=$($sbatch_cmd "$job_script")
        echo "  [RUN] $step_name -> Job ID: $job_id" >&2
        echo "$job_id"
    fi
}

echo "[1/9] Training SFT control modules..."
JOB_SFT=$(check_and_run "SFT Training" "models/sft_llama3.1_8b_sentiment_seed${SEED}/checkpoint-400" "jobs/sft_train.job" "")

echo "[2/9] Base model inference..."
# Base model inference does not depend on SFT
JOB_INF_BASE=$(check_and_run "Base Inference" "outputs/base_llama31_sentiment_seed${SEED}.json" "jobs/inference_base_models.job" "")

echo "[3/9] Base model evaluation..."
JOB_EVAL_BASE=$(check_and_run "Base Eval" "predictions/base_llama31_sentiment_seed${SEED}.json" "jobs/evaluate_base_models.job" "$JOB_INF_BASE")

echo "[4/9] Identity inference..."
JOB_INF_ID=$(check_and_run "Identity Inference" "outputs/identity_llama31_sentiment_seed${SEED}.json" "jobs/inference_identity_sentiment.job" "$JOB_SFT")

echo "[5/9] Identity evaluation..."
JOB_EVAL_ID=$(check_and_run "Identity Eval" "predictions/identity_llama31_sentiment_seed${SEED}.json" "jobs/evaluate_identity.job" "$JOB_INF_ID")

echo "[6/9] Zero-shot ported module inference..."
JOB_INF_PORT=$(check_and_run "Porting Inference" "outputs/llama31_to_llama32_sentiment_seed${SEED}.json" "jobs/inference_llama31_module_sentiment.job" "$JOB_SFT")

echo "[7/9] Zero-shot ported module evaluation..."
JOB_EVAL_PORT=$(check_and_run "Porting Eval" "predictions/llama31_to_llama32_sentiment_seed${SEED}.json" "jobs/evaluate_llama31_module.job" "$JOB_INF_PORT")

echo "[8/9] Few-step adaptation..."
JOB_FEW_STEP=$(check_and_run "Few-step Train" "results/few_step_llama31_to_llama32_sentiment_seed${SEED}.json" "jobs/train_post_porting_sft.job" "$JOB_SFT")

echo "[9/9] Few-step evaluation..."
JOB_EVAL_FEW_STEP=$(check_and_run "Few-step Eval" "predictions/few_step_llama31_to_llama32_sentiment_seed${SEED}.json" "jobs/evaluate_few_step.job" "$JOB_FEW_STEP")

echo "=========================================================="
echo "Reproduction pipeline check complete."
echo "Track active jobs with: squeue -u $USER"
echo "=========================================================="
