#!/bin/bash
# analyse.sh - Run global evaluation, metric consolidation and summary reports

echo "Submitting Consolidated Analysis Job..."
mkdir -p logs results outputs

if [ -f "jobs/analyze.job" ]; then
    JOB_ANALYZE=$(sbatch --parsable jobs/analyze.job)
    echo "Submitted analyze job: Job ID $JOB_ANALYZE"
else
    echo "Running analysis directly via Python..."
    python3 scripts/analyze_results.py
fi

echo "=========================================================="
echo "Analysis initiated. Outputs will be saved in results/."
echo "=========================================================="
