#!/bin/bash
# run_rl.sh - Reinforcement Learning Pipeline (V1, V2, V2.1, V3, V4)

echo "Submitting RL Training and Evaluation Pipeline..."
mkdir -p logs outputs predictions results models

# Function to submit a training job and chain its inference evaluation
submit_rl_stage() {
    local version=$1
    local train_job=$2
    local eval_job=$3

    if [ -f "$train_job" ]; then
        echo "Launching RL Training ($version)..."
        JOB_TRAIN=$(sbatch --parsable $train_job)
        echo "  -> Train Job ID: $JOB_TRAIN"

        if [ -f "$eval_job" ]; then
            JOB_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN $eval_job)
            echo "  -> Chained Eval Job ID: $JOB_EVAL"
        fi
    else
        echo "Warning: $train_job not found, skipping $version."
    fi
}

# 1. RL Version 1
submit_rl_stage "v1" "jobs/rl_train_v1.job" "jobs/inference_rl_v1_sentiment.job"

# 2. RL Version 2
submit_rl_stage "v2" "jobs/rl_train_v2.job" "jobs/inference_rl_v2_sentiment.job"

# 3. RL Version 2.1 (Length-Normalized Contrast)
submit_rl_stage "v2_1" "jobs/rl_train_v2_1.job" "jobs/inference_rl_v2_1_sentiment.job"

# 4. RL Version 3 (SLOR Integration)
submit_rl_stage "v3" "jobs/rl_train_v3.job" "jobs/inference_rl_v3_sentiment.job"

# 5. RL Version 4 (Shannon Entropy Diversity)
submit_rl_stage "v4" "jobs/rl_train_v4.job" "jobs/inference_rl_v4_sentiment.job"

echo "=========================================================="
echo "All RL training & evaluation jobs scheduled."
echo "Track status with: squeue -u $USER"
echo "=========================================================="
