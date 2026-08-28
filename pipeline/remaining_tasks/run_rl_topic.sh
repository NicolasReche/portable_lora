#!/bin/bash
# run_rl_topic.sh
# Launches all remaining or modified jobs for Topic

echo "=========================================================="
echo " 0. DICTIONARY PREPARATION (SLOR)"
echo "=========================================================="
python scripts/compute_unigrams.py

echo ""
echo "=========================================================="
echo " 1. TRAINING (To be launched now)"
echo "=========================================================="
echo "--- TOPIC (ALL VERSIONS : V1 -> V4.1) ---"
for v in "v1" "v2" "v2_1" "v3" "v3_1" "v4" "v4_1"; do
    sbatch jobs/train_rl/topic/rl_train_${v}_llama31.job
    sbatch jobs/train_rl/topic/rl_train_${v}_llama32.job
done

echo ""
echo "=========================================================="
echo " 2. INFERENCE (Zero-Shot & Identity) - TO BE LAUNCHED AFTER TRAINING"
echo "=========================================================="
echo "--- TOPIC ---"
for v in "v1" "v2" "v2_1" "v3" "v3_1" "v4" "v4_1"; do
    echo "sbatch jobs/inference_rl/topic/inference_rl_${v}_llama31_module.job"
    echo "sbatch jobs/inference_rl/topic/inference_rl_${v}_llama32_module.job"
done

echo ""
echo "=========================================================="
echo " 3. FEW-STEP ADAPTATION - TO BE LAUNCHED AFTER TRAINING"
echo "=========================================================="
echo "--- TOPIC ---"
for v in "v1" "v2" "v2_1" "v3" "v3_1" "v4" "v4_1"; do
    echo "sbatch jobs/few_step_rl.job ${v} all all topic 7097"
done
echo "=========================================================="
