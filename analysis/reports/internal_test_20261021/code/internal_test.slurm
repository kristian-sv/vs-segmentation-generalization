#!/usr/bin/env bash
# Internal-test evaluation (PROTOCOL.md). From the repository root:
# CPU: sbatch analysis/reports/internal_test_20261021/code/internal_test.slurm preflight|build|score|summary
# GPU: sbatch --array=1-10 --partition=accel --gpus=1 --time=01:00:00 analysis/reports/internal_test_20261021/code/internal_test.slurm predict
#      tasks 1-4 = followup_checkpoints.csv rows (stress inputs, as stored and re-stacked); 5-10 = runs.csv rows 1-6 (manual, unchanged scans)
#SBATCH --job-name=internal-test
#SBATCH --account=nn11127k
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/internal-test-%A_%a.log
set -euo pipefail
cd /cluster/projects/nn11127k/ksv023/thesis
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
r=analysis/reports/internal_test_20261021
earlier=analysis/reports/model_insight_20261012
if [[ "$1" == predict ]]; then
    export T1_MVP_GPU=1 nnUNet_extTrainer="$PWD/experiments/nnunet_trainers"
    mkdir -p "$r/provenance"
    git rev-parse HEAD > "$r/provenance/code_commit_task${SLURM_ARRAY_TASK_ID}.txt"
    set -x
    if [[ "$SLURM_ARRAY_TASK_ID" -le 4 ]]; then
        # Same predictor settings as the step-A/B follow-up predictions (model_insight.slurm followup-predict).
        IFS=, read -r run model checkpoint_sha < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$earlier/followup_checkpoints.csv")
        printf '%s  %s\n' "$checkpoint_sha" "$model/fold_0/checkpoint_best.pth" | sha256sum --check --status
        [[ -f "$r/stress_test/manifest.csv" && ! -e "$r/stress_test/predictions/$run" && ! -e "$r/stress_test/predictions_lps/${run}_restacked" ]]
        mkdir -p "$r/stress_test/predictions" "$r/stress_test/predictions_lps"
        for input in stored restacked; do
            [[ "$input" == stored ]] && dest="$r/stress_test/predictions/$run" || dest="$r/stress_test/predictions_lps/${run}_restacked"
            srun bash experiments/environment.sh nnunet nnUNetv2_predict_from_modelfolder -i "$r/stress_test/nnunet_input/$input" \
                -o "$dest" -m "$model" -f 0 -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2
        done
    else
        # Same exporter and invocation as the September evaluation (experiments/evaluate.slurm), unchanged scans only.
        IFS=, read -r run family seed checkpoint checkpoint_sha < <(sed -n "$((SLURM_ARRAY_TASK_ID - 3))p" analysis/reports/t1_evaluation_20260912/runs.csv)
        [[ "$family" != nnunet ]]
        printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check --status
        [[ ! -e "$r/original_runs/$run" ]]
        mkdir -p "$r/original_runs/$run"
        srun bash experiments/environment.sh manual python -u src/evaluate.py manual-export \
            --manifest "$r/test_manifest.csv" --manual "$family" --seed "$seed" \
            --checkpoint "$checkpoint" --output "$r/original_runs/$run/predictions"
    fi
    exit 0
fi
module --force purge
module load NRIS/CPU SciPy-bundle/2024.05-gfbf-2024a  # x86 CPU nodes, as model_insight.slurm
if [[ "$1" == preflight ]]; then
    python -u "$r/code/preflight.py"
else
    python -u "$r/code/internal_test.py" "$@"
fi
