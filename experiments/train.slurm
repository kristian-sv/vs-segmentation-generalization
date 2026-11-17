#!/usr/bin/env bash
# One fresh run per submission. See README.md; this file is never submitted by setup.
#SBATCH --job-name=t1-comparison
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/t1-%j.log
set -euo pipefail
cd /cluster/projects/nn11127k/ksv023/thesis
export T1_MVP_GPU=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
# Keep experiment records locally; syncing is a separate user action.
export WANDB_MODE=offline
case "${1:-}" in
    manual)
        [[ $# == 3 ]] || { echo 'Usage: train.slurm manual baseline|dropout01 42|43|44' >&2; exit 2; }
        srun bash experiments/environment.sh manual \
            python -u experiments/run.py --manual "$2" --seed "$3" --execute
        ;;
    nnunet)
        [[ $# == 1 ]] || { echo 'Usage: train.slurm nnunet' >&2; exit 2; }
        srun bash experiments/environment.sh nnunet \
            python -u experiments/run.py --nnunet --execute
        ;;
    nnunet-cv)
        [[ $# == 1 && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-4]$ ]] || {
            echo 'Usage: sbatch --array=0-4 --time=4-00:00:00 train.slurm nnunet-cv' >&2; exit 2;
        }
        srun bash experiments/environment.sh nnunet \
            python -u experiments/nnunet_prepare.py --cv-fold "$SLURM_ARRAY_TASK_ID" --execute
        srun bash experiments/environment.sh nnunet \
            python -u experiments/run.py --nnunet --cv-fold "$SLURM_ARRAY_TASK_ID" --execute
        ;;
    nnunet-followup)
        # Preregistered follow-up (analysis/reports/model_insight_20261012/PREREGISTRATION.md)
        [[ $# == 3 ]] || { echo 'Usage: train.slurm nnunet-followup nnUNetTrainer|nnUNetTrainerSlabThick 42|43' >&2; exit 2; }
        srun bash experiments/environment.sh nnunet \
            python -u experiments/run.py --nnunet --trainer "$2" --seed "$3" --execute
        ;;
    nnunet-smoke)
        # A few epochs plus sample patches for review; never a result. T1_SMOKE_EPOCHS overrides the default 3.
        srun bash experiments/environment.sh nnunet \
            python -u experiments/run.py --nnunet --trainer nnUNetTrainerSlabThick --smoke --execute
        ;;
    *) echo 'Expected manual, nnunet, nnunet-cv, nnunet-followup or nnunet-smoke. See README.md.' >&2; exit 2 ;;
esac
