#!/bin/bash
# run_rl.sh - Reinforcement Learning Pipeline (V1, V2, V2.1, V3, V4)

echo "Submitting RL Training and Evaluation Pipeline (Sentiment)..."
mkdir -p logs outputs predictions results models

SEED=7097

submit_rl_model() {
    local version=$1
    local model_short=$2
    local model_long=$3

    local train_job="jobs/train_rl/sentiment/rl_train_${version}_${model_short}.job"
    local inf_job="jobs/inference_rl/sentiment/inference_rl_${version}_${model_short}_module.job"
    
    local model_dir="models/rl_${model_long}_sentiment_${version}_seed${SEED}"
    
    local out1="outputs/rl_${version}_${model_short}_to_llama31_sentiment_seed${SEED}.json"
    local out2="outputs/rl_${version}_${model_short}_to_llama32_sentiment_seed${SEED}.json"

    # 1. Check training status
    JOB_TRAIN=""
    if [ -d "$model_dir" ]; then
        echo "  [SKIP] Training already done for $version ($model_short). Model directory exists."
    else
        if [ -f "$train_job" ]; then
            echo "  [RUN] Launching RL training ($version for $model_short)..."
            JOB_TRAIN=$(sbatch --parsable $train_job)
            echo "    -> Train Job ID: $JOB_TRAIN"
        else
            echo "  [ERROR] Script $train_job not found."
        fi
    fi

    # 2. Check inference status
    if [ -f "$out1" ] && [ -f "$out2" ]; then
        echo "  [SKIP] Inference already done for $version ($model_short). Results exist."
    else
        if [ -f "$inf_job" ]; then
            if [ -n "$JOB_TRAIN" ]; then
                # If training was just launched, wait for it to finish
                echo "  [RUN] Launching inference ($version for $model_short) WAITING for Job $JOB_TRAIN..."
                JOB_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN $inf_job)
                echo "    -> Chained Inf/Eval Job ID: $JOB_EVAL"
            else
                # If training was already done (SKIP) but inference is missing, launch it immediately
                echo "  [RUN] Immediate launch of inference ($version for $model_short)..."
                JOB_EVAL=$(sbatch --parsable $inf_job)
                echo "    -> Independent Inf/Eval Job ID: $JOB_EVAL"
            fi
        else
            echo "  [ERROR] Script $inf_job not found."
        fi
    fi
}

for VERSION in "v1" "v2" "v2_1" "v3" "v4"; do
    echo "--- Version $VERSION ---"
    submit_rl_model "$VERSION" "llama31" "llama3.1_8b"
    submit_rl_model "$VERSION" "llama32" "llama3.2_3b"
    echo ""
done

echo "=========================================================="
echo "RL pipeline check and execution completed"
echo "Track running tasks with: squeue -u $USER"
echo "=========================================================="
