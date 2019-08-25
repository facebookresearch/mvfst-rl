#!/bin/bash

# Usage: sbatch slurm_launch.sh [args]

#SBATCH --job-name=mvrlfst
#SBATCH --output=/checkpoint/%u/logs/mvrlfst-%j.out
#SBATCH --error=/checkpoint/%u/logs/mvrlfst-%j.err
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=5:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append

# TODO (viswanath): hyperparam sweep

# ArgumentParser
CONDA_ENV=""
NUM_ACTORS=0
MAX_JOBS=0
JOB_IDS=""
LOG_DIR=""
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

srun --label "$PWD"/scripts/slurm_wrapper.sh \
     --conda_env "$CONDA_ENV" \
     --num_actors "$NUM_ACTORS" \
     --max_jobs "$MAX_JOBS" \
     --job_ids "$JOB_IDS" \
     --logdir "$LOG_DIR" \
