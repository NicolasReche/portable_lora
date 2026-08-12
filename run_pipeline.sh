#!/bin/bash
# run_pipeline.sh

echo "Soumission de la pipeline d'expérimentation complète (Sentiment)..."
mkdir -p logs outputs predictions results models

echo "[1/9] Training of sentiment control modules (SFT)"
JOB_SFT=$(sbatch --parsable jobs/sft_train.job)

echo "[2/9] Inference of raw models"
JOB_INF_BASE=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/inference_base_models.job)

echo "[3/9] Inference of identity (module on its own model)"
JOB_INF_ID=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/inference_identity_sentiment.job)

echo "[4/9] Few-step portability (Llama3.1 <-> Llama3.2)"
JOB_PORT=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/train_post_porting_sft.job)

echo "[5/9] Base Model Evaluation"
JOB_EVAL_BASE=$(sbatch --parsable --dependency=afterok:$JOB_INF_BASE jobs/evaluate_base_models.job)

echo "[6/9] Ported Module Inference (Sentiment)"
JOB_INF_PORT=$(sbatch --parsable --dependency=afterok:$JOB_PORT jobs/inference_llama31_module_sentiment.job)

echo "[7/9] Ported Module Evaluation"
JOB_EVAL_PORT=$(sbatch --parsable --dependency=afterok:$JOB_INF_PORT jobs/evaluate_llama31_module.job)

echo "[8/9] RL Training"
JOB_RL=$(sbatch --parsable --dependency=afterok:$JOB_SFT jobs/rl_train.job)

echo "[9/9] Analysis"
JOB_ANALYZE=$(sbatch --parsable --dependency=afterok:$JOB_EVAL_BASE:$JOB_EVAL_PORT:$JOB_RL jobs/analyze.job)

echo "=========================================================="
echo "Tout est soumis ! Vous pouvez fermer votre terminal ou "
echo "mettre votre PC en veille. Utilisez 'squeue -u \$USER' "
echo "pour suivre l'avancement."
echo "=========================================================="
