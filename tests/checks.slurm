#!/usr/bin/env bash
# Bounded verification after cleanup, never a full experiment.
#SBATCH --job-name=t1-cleanup-check
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=tests/work/checks-%j.log
set -euo pipefail
cd /cluster/projects/nn11127k/ksv023/thesis
export T1_MVP_GPU=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=4 WANDB_MODE=disabled
srun bash experiments/environment.sh manual python -u tests/check_workflow.py \
    --real --manual-cli-smoke --output "tests/work/scoring-${SLURM_JOB_ID}.json"
srun bash experiments/environment.sh nnunet python -u \
    tests/check_nnunet.py "tests/work/nnunet-${SLURM_JOB_ID}.json"
