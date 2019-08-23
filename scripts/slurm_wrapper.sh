#!/bin/bash

# Usage: ./slurm_wrapper.sh [args]

CUR_DIR=$(dirname "$(realpath -s "$0")")
ROOT_DIR="$CUR_DIR"/..

# ArgumentParser
CONDA_ENV=""
NUM_ACTORS=0
MAX_JOBS=0
JOB_IDS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --conda_env )
      CONDA_ENV="$2"
      shift 2;;
    --num_actors )
      NUM_ACTORS="$2"
      shift 2;;
    --max_jobs )
      MAX_JOBS="$2"
      shift 2;;
    --job_ids )
      JOB_IDS="$2"
      shift 2;;
    --logdir )
      LOG_DIR="$2"
      shift 2;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

if [ -z "$CONDA_ENV" ]; then
  echo "--conda_env must be specified."
  exit 1
fi

LOG_DIR=${LOG_DIR:-"/checkpoint/$USER/mvrlfst/$SLURM_JOB_ID"}
mkdir -p "$LOG_DIR"

echo "Conda env: $CONDA_ENV"
echo "Num actors: $NUM_ACTORS"
echo "Max jobs: $MAX_JOBS"
echo "Job ids: $JOB_IDS"
echo "Log dir: $LOG_DIR"


. /usr/share/modules/init/sh
source deactivate
module purge
module load anaconda3
module load cuda/9.2
module load cudnn/v7.3-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2
source activate "$HOME"/.conda/envs/"$CONDA_ENV"

export CUDA_HOME="/public/apps/cuda/9.2"
export CUDNN_INCLUDE_DIR="/public/apps/cudnn/v7.3/cuda/include"
export CUDNN_LIB_DIR="/public/apps/cudnn/v7.3/cuda/lib64"

echo "Running job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

export PYTHONPATH="$PYTHONPATH":"$ROOT_DIR"
export PYTHONUNBUFFERED=True

"$ROOT_DIR"/scripts/train.sh \
  --num_actors "$NUM_ACTORS" \
  --max_jobs "$MAX_JOBS" \
  --job_ids "$JOB_IDS" \
  --logdir "$LOG_DIR"
