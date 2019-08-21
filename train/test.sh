#!/bin/bash -e

# Usage: ./test.sh --checkpoint <checkpoint.tar>
#                  [--max_jobs M] [--job_ids 0,1,2]
#                  [--test_runs_per_job N]

# ArgumentParser
CHECKPOINT=""
MAX_JOBS=0
JOB_IDS=""
TEST_RUNS_PER_JOB=3
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --checkpoint )
      CHECKPOINT="$2"
      shift 2;;
    --max_jobs )
      MAX_JOBS="$2"
      shift 2;;
    --job_ids )
      JOB_IDS="$2"
      shift 2;;
    --test_runs_per_job )
      TEST_RUNS_PER_JOB="$2"
      shift 2;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

if [ -z "$CHECKPOINT" ]; then
  echo "--checkpoint must be specified"
  exit 1
fi

echo "CHECKPOINT: $CHECKPOINT"
echo "MAX_JOBS: $MAX_JOBS"
echo "JOB_IDS: $JOB_IDS"
echo "TEST_RUNS_PER_JOB: $TEST_RUNS_PER_JOB"

CUR_DIR=$(dirname "$(realpath -s "$0")")
ROOT_DIR="$CUR_DIR"/..
TORCHBEAST_DIR="$ROOT_DIR"/third-party/torchbeast

LOG_DIR="$CUR_DIR/logs/test"
mkdir -p $LOG_DIR

module unload cuda
module unload cudnn
module unload NCCL
module load cuda/9.2
module load cudnn/v7.3-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2

export CUDA_HOME="/public/apps/cuda/9.2"
export CUDNN_INCLUDE_DIR="/public/apps/cudnn/v7.3/cuda/include"
export CUDNN_LIB_DIR="/public/apps/cudnn/v7.3/cuda/lib64"

PYTHONPATH=$PYTHONPATH:"$TORCHBEAST_DIR"

# Unix domain socket path for RL server address
SOCKET_PATH="/tmp/rl_server_path"
rm -f $SOCKET_PATH

PANTHEON_LOG_DIR="$LOG_DIR/pantheon"
mkdir -p $PANTHEON_LOG_DIR
PANTHEON_LOG="$PANTHEON_LOG_DIR/pantheon.log"
TEST_LOG="$LOG_DIR/polybeast.log"

# For testing, pantheon_env.py decides termination, so launch polybeast.py first
# in the background and kill it once pantheon_env.py returns.
# TODO (viswanath): More params
PYTHONPATH=$PYTHONPATH OMP_NUM_THREADS=1 python3 $ROOT_DIR/train/polybeast.py \
  --mode=test \
  --address "unix:$SOCKET_PATH" \
  --checkpoint "$CHECKPOINT" \
  --disable_cuda \
  > "$TEST_LOG" 2>&1 &
BG_PID=$!
echo "Polybeast running in background (pid: $BG_PID), logfile: $TEST_LOG."

# Now start pantheon_env.py
echo "Starting pantheon, logfile: $PANTHEON_LOG."
python3 $ROOT_DIR/train/pantheon_env.py \
  --mode=test \
  --max_jobs "$MAX_JOBS" \
  --job_ids "$JOB_IDS" \
  --test_runs_per_job "$TEST_RUNS_PER_JOB" \
  -v 1 \
  --logdir "$PANTHEON_LOG_DIR" \
  > "$PANTHEON_LOG" 2>&1

# Interrupt the background polybeast process.
echo "Done testing, terminating polybeast."
kill -INT "$BG_PID"
