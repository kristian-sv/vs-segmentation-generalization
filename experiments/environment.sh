#!/usr/bin/env bash
# Existing software only. Invoke with: bash .../environment.sh manual|nnunet COMMAND...
# Example CPU preflight: environment.sh nnunet python -c 'import nnunetv2'
# For future GPU execution in an allocation, set T1_MVP_GPU=1 explicitly.
set -euo pipefail
kind="$1"
shift
container=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
overlay=/cluster/projects/nn11127k/ksv023/mmiv-nnunet/containers/nnunet-notebook-pydeps.img
root=/cluster/home/ksv023/thesis
gpu_args=()
if [[ "${T1_MVP_GPU:-0}" == 1 ]]; then gpu_args=(--nv); fi
case "$kind" in
  manual)
    exec apptainer exec "${gpu_args[@]}" --bind /cluster:/cluster \
      --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 \
      --env PYTHONPATH="$root/.pip_arm64:$root/src" \
      --env LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/.singularity.d/libs \
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs "$container" "$@"
    ;;
  nnunet)
    exec apptainer exec "${gpu_args[@]}" --overlay "$overlay:ro" --bind /cluster:/cluster \
      --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUSERBASE=/home/apptainer/user_software \
      --env PATH=/home/apptainer/user_software/bin:/usr/local/bin:/usr/bin:/bin \
      --env nnUNet_raw="$root/experiments/work/nnUNet_raw" \
      --env nnUNet_preprocessed="$root/experiments/work/nnUNet_preprocessed" \
      --env nnUNet_results="$root/outputs/t1/nnunet" \
      --env LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/.singularity.d/libs \
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
      "$container" "$@"
    ;;
  *) echo 'Expected manual or nnunet' >&2; exit 2 ;;
esac
