#!/usr/bin/env bash
# Evaluate the seven frozen checkpoints: sbatch --array=1-7 evaluate.slurm validation|mcrc
#SBATCH --job-name=t1-evaluate
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/t1-eval-%A_%a.log
set -euo pipefail
cd /cluster/projects/nn11127k/ksv023/thesis
export T1_MVP_GPU=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == vsseg-review ]]; then
    srun bash experiments/environment.sh nnunet \
        python -u experiments/predict_review.py --execute
    exit 0
fi
cohort=${1:?Expected validation or mcrc}
[[ "$cohort" == validation || "$cohort" == mcrc ]]
[[ "${SLURM_ARRAY_TASK_ID:-0}" -ge 1 && "$SLURM_ARRAY_TASK_ID" -le 7 ]]
record=analysis/reports/t1_evaluation_20260912
manifest="$record/${cohort}_manifest.csv"
IFS=, read -r run family seed checkpoint checkpoint_sha < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$record/runs.csv")
printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check --status
output="outputs/t1/evaluation_20260912/$cohort/$run"
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
mkdir -p "$output"
cp "$0" "$record/runs.csv" "$manifest" "$output/"
git rev-parse HEAD > "$output/code_commit.txt"
set -x
if [[ "$family" == nnunet ]]; then
    # Validate the same manifest and stage only its MRI images for nnU-Net.
    srun bash experiments/environment.sh manual python - "$manifest" "$output/input" <<'PY'
import sys
from pathlib import Path
from evaluate import read_manifest, pair_geometry
rows = read_manifest(sys.argv[1])
directory = Path(sys.argv[2])
directory.mkdir()
for row in rows:
    pair_geometry(row)
    (directory / (row['case_id'] + '_0000.nii.gz')).symlink_to(row['image'])
PY
    srun bash experiments/environment.sh nnunet nnUNetv2_predict \
        -i "$output/input" -o "$output/predictions" -d 901 -c 3d_fullres -f 0 \
        -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2
else
    srun bash experiments/environment.sh manual python -u src/evaluate.py manual-export \
        --manifest "$manifest" --manual "$family" --seed "$seed" \
        --checkpoint "$checkpoint" --output "$output/predictions"
fi
srun bash experiments/environment.sh manual python -u src/evaluate.py score \
    --manifest "$manifest" --predictions "$output/predictions" --output "$output/scores"
