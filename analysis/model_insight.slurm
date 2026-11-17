#!/usr/bin/env bash
# CPU: sbatch analysis/model_insight.slurm orientation|reorient-stage|reorient-restore|cases|bundle|followup-figures|followup-posthoc|followup-fp-figure
# GPU: sbatch --partition=accel --gpus=1 --time=00:20:00 analysis/model_insight.slurm nnunet-reoriented
#      sbatch --array=1-7 --partition=accel --gpus=1 --time=01:00:00 analysis/model_insight.slurm stress-predict
#SBATCH --job-name=model-insight
#SBATCH --account=nn11127k
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/model-insight-%j.log
set -euo pipefail
cd /cluster/projects/nn11127k/ksv023/thesis
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
out=analysis/reports/model_insight_20261012/nnunet_reoriented
if [[ "$1" == nnunet-reoriented ]]; then
    # Same checkpoint and invocation as the September MC-RC evaluation (no TTA); only the input axis order differs.
    IFS=, read -r run family seed checkpoint checkpoint_sha < <(grep '^nnunet_s42,' analysis/reports/t1_evaluation_20260912/runs.csv)
    printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check --status
    [[ -d "$out/input" && ! -e "$out/predictions_lps" ]]
    git rev-parse HEAD > "$out/code_commit.txt"
    export T1_MVP_GPU=1
    srun bash experiments/environment.sh nnunet python -c 'import inspect, nnunetv2; from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO as R; print(nnunetv2.__file__); print(inspect.getsource(R.read_images))'
    set -x
    srun bash experiments/environment.sh nnunet nnUNetv2_predict -i "$out/input" -o "$out/predictions_lps" \
        -d 901 -c 3d_fullres -f 0 -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2
    exit 0
fi
if [[ "$1" == stress-predict ]]; then
    # Array task i = runs.csv row i (1-6 manual, 7 nnU-Net); same exporter/invocation as the September evaluation.
    st=analysis/reports/model_insight_20261012/stress_test
    IFS=, read -r run family seed checkpoint checkpoint_sha < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" analysis/reports/t1_evaluation_20260912/runs.csv)
    printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check --status
    [[ -f "$st/manifest.csv" && ! -e "$st/predictions/$run" ]]
    mkdir -p "$st/predictions" "$st/predictions_lps"
    export T1_MVP_GPU=1
    set -x
    if [[ "$family" == nnunet ]]; then
        for input in stored restacked; do
            [[ "$input" == stored ]] && dest="$st/predictions/$run" || dest="$st/predictions_lps/${run}_restacked"
            srun bash experiments/environment.sh nnunet nnUNetv2_predict -i "$st/nnunet_input/$input" -o "$dest" \
                -d 901 -c 3d_fullres -f 0 -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2
        done
    else
        srun bash experiments/environment.sh manual python -u src/evaluate.py manual-export \
            --manifest "$st/manifest.csv" --manual "$family" --seed "$seed" \
            --checkpoint "$checkpoint" --output "$st/predictions/$run"
    fi
    exit 0
fi
if [[ "$1" == followup-predict ]]; then
    # Preregistered evaluation (PREREGISTRATION.md). Task i = row i of followup_checkpoints.csv.
    # $2 = stress (tasks 2-4; plain s42 stress predictions exist as nnunet_s42) or mcrc (tasks 1-4, re-stacked inputs).
    r=analysis/reports/model_insight_20261012
    IFS=, read -r run model checkpoint_sha < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$r/followup_checkpoints.csv")
    printf '%s  %s\n' "$checkpoint_sha" "$model/fold_0/checkpoint_best.pth" | sha256sum --check --status
    export T1_MVP_GPU=1 nnUNet_extTrainer="$PWD/experiments/nnunet_trainers"
    predict() {  # same predictor settings as the September nnUNetv2_predict call; -m names the model folder
        srun bash experiments/environment.sh nnunet nnUNetv2_predict_from_modelfolder -i "$1" -o "$2" \
            -m "$model" -f 0 -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2
    }
    set -x
    if [[ "$2" == stress ]]; then
        [[ "$run" != plain_s42 && ! -e "$r/stress_test/predictions/nnunet_$run" ]]
        predict "$r/stress_test/nnunet_input/stored" "$r/stress_test/predictions/nnunet_$run"
        predict "$r/stress_test/nnunet_input/restacked" "$r/stress_test/predictions_lps/nnunet_${run}_restacked"
    else
        [[ "$2" == mcrc && ! -e "$r/mcrc_followup/predictions_lps/$run" ]]
        mkdir -p "$r/mcrc_followup/predictions_lps"
        predict "$r/nnunet_reoriented/input" "$r/mcrc_followup/predictions_lps/$run"
    fi
    exit 0
fi
module --force purge
if [[ "$1" == stress-summary || "$1" == followup-training ]]; then
    module load NRIS/CPU matplotlib/3.10.3-gfbf-2025a  # plotting only; nibabel is not needed
elif [[ "$1" == followup-figures || "$1" == followup-posthoc || "$1" == followup-fp-figure ]]; then
    module load NRIS/CPU matplotlib/3.10.3-gfbf-2025a  # Python 3.13: expose only the pure-Python user-site nibabel
    mkdir -p logs/py313_nibabel && ln -sfn ~/.local/lib/python3.12/site-packages/nibabel logs/py313_nibabel/nibabel
    export PYTHONPATH="$PWD/logs/py313_nibabel${PYTHONPATH:+:$PYTHONPATH}"
else
    module load NRIS/CPU SciPy-bundle/2024.05-gfbf-2024a  # x86 CPU nodes
fi
python -u analysis/model_insight.py "$@"
