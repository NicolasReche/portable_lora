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

    # 1. Vérification de l'entraînement
    JOB_TRAIN=""
    if [ -d "$model_dir" ]; then
        echo "  [SKIP] Entraînement déjà fait pour $version ($model_short). Le dossier du modèle existe."
    else
        if [ -f "$train_job" ]; then
            echo "  [RUN] Lancement de l'entraînement RL ($version pour $model_short)..."
            JOB_TRAIN=$(sbatch --parsable $train_job)
            echo "    -> Train Job ID: $JOB_TRAIN"
        else
            echo "  [ERROR] Script $train_job introuvable."
        fi
    fi

    # 2. Vérification de l'inférence
    if [ -f "$out1" ] && [ -f "$out2" ]; then
        echo "  [SKIP] Inférence déjà faite pour $version ($model_short). Les résultats existent."
    else
        if [ -f "$inf_job" ]; then
            if [ -n "$JOB_TRAIN" ]; then
                # Si l'entraînement vient d'être lancé, on attend qu'il finisse
                echo "  [RUN] Lancement de l'inférence ($version pour $model_short) EN ATTENTE du Job $JOB_TRAIN..."
                JOB_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN $inf_job)
                echo "    -> Chained Inf/Eval Job ID: $JOB_EVAL"
            else
                # Si l'entraînement était déjà fini (SKIP) mais que l'inférence manque, on la lance tout de suite
                echo "  [RUN] Lancement immédiat de l'inférence ($version pour $model_short)..."
                JOB_EVAL=$(sbatch --parsable $inf_job)
                echo "    -> Independent Inf/Eval Job ID: $JOB_EVAL"
            fi
        else
            echo "  [ERROR] Script $inf_job introuvable."
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
echo "Vérification et exécution du pipeline RL terminées."
echo "Suivez les tâches en cours avec: squeue -u $USER"
echo "=========================================================="
