#!/bin/bash
# run_rl_sentiment.sh
# Launches all remaining or modified jobs for Sentiment

echo "=========================================================="
echo " 0. DICTIONARY PREPARATION (SLOR)"
echo "=========================================================="
python scripts/compute_unigrams.py

echo ""
echo "=========================================================="
echo " 1. TRAINING (To be launched now)"
echo "=========================================================="
echo "--- SENTIMENT (V3, V3.1, V4, V4.1) ---"
sbatch jobs/train_rl/sentiment/rl_train_v3_llama31.job
sbatch jobs/train_rl/sentiment/rl_train_v3_llama32.job
sbatch jobs/train_rl/sentiment/rl_train_v3_1_llama31.job
sbatch jobs/train_rl/sentiment/rl_train_v3_1_llama32.job
sbatch jobs/train_rl/sentiment/rl_train_v4_llama31.job
sbatch jobs/train_rl/sentiment/rl_train_v4_llama32.job
sbatch jobs/train_rl/sentiment/rl_train_v4_1_llama31.job
sbatch jobs/train_rl/sentiment/rl_train_v4_1_llama32.job

echo ""
echo "=========================================================="
echo " 2. INFERENCE (Zero-Shot & Identity) - TO BE LAUNCHED AFTER TRAINING"
echo "=========================================================="
echo "--- SENTIMENT ---"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v3_llama31_module.job"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v3_llama32_module.job"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v3_1_llama31_module.job"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v3_1_llama32_module.job"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v4_llama31_module.job"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v4_llama32_module.job"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v4_1_llama31_module.job"
echo "sbatch jobs/inference_rl/sentiment/inference_rl_v4_1_llama32_module.job"

echo ""
echo "=========================================================="
echo " 3. FEW-STEP ADAPTATION - TO BE LAUNCHED AFTER TRAINING"
echo "=========================================================="
echo "--- SENTIMENT ---"
echo "sbatch jobs/few_step_rl.job v1 llama31 llama32 sentiment 7097"
echo "sbatch jobs/few_step_rl.job v2_1 llama32 llama31 sentiment 7097"
echo "sbatch jobs/few_step_rl.job v3 all all sentiment 7097"
echo "sbatch jobs/few_step_rl.job v3_1 all all sentiment 7097"
echo "sbatch jobs/few_step_rl.job v4 all all sentiment 7097"
echo "sbatch jobs/few_step_rl.job v4_1 all all sentiment 7097"
echo "=========================================================="
