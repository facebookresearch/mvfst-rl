#!/bin/bash -e

# Usage: ./train.sh [--num_actors N] [--max_jobs M] [--job_ids 0,1,2]
#                   [--logdir /log/dir]

CUR_DIR=$(dirname "$(realpath -s "$0")")
ROOT_DIR="$CUR_DIR"/..

# ArgumentParser
NUM_ACTORS=0
MAX_JOBS=0
JOB_IDS=""
BASE_LOG_DIR="$CUR_DIR/logs"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
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
      BASE_LOG_DIR="$2"
      shift 2;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

echo "NUM_ACTORS: $NUM_ACTORS"
echo "MAX_JOBS: $MAX_JOBS"
echo "JOB_IDS: $JOB_IDS"
echo "BASE_LOG_DIR: $BASE_LOG_DIR"

LOG_DIR="$BASE_LOG_DIR/train"
mkdir -p $LOG_DIR

TORCHBEAST_DIR="$ROOT_DIR"/third-party/torchbeast
PYTHONPATH=$PYTHONPATH:"$TORCHBEAST_DIR"

# Unix domain socket path for RL server address
SOCKET_PATH="/tmp/rl_server_path"
rm -f $SOCKET_PATH

POLYBEAST_LOG="$LOG_DIR/polybeast.log"
PANTHEON_LOG="$LOG_DIR/pantheon.log"
PANTHEON_LOG_DIR="$LOG_DIR/pantheon"
mkdir -p $PANTHEON_LOG_DIR

CHECKPOINT="$LOG_DIR/checkpoint.tar"

# Start pantheon_env.py in the background
python3 $ROOT_DIR/train/pantheon_env.py \
  --mode=train \
  --num_actors "$NUM_ACTORS" \
  --max_jobs "$MAX_JOBS" \
  --job_ids "$JOB_IDS" \
  -v 1 \
  --logdir "$PANTHEON_LOG_DIR" \
  > "$PANTHEON_LOG" 2>&1 &
BG_PID=$!
echo "Pantheon running in background (pid: $BG_PID), logfile: $PANTHEON_LOG."

# Start the trainer
# TODO (viswanath): More params
echo "Starting polybeast, logfile: $POLYBEAST_LOG."
echo "Checkpoint: $CHECKPOINT."
PYTHONPATH=$PYTHONPATH OMP_NUM_THREADS=1 python3 $ROOT_DIR/train/polybeast.py \
  --mode=train \
  --address "unix:$SOCKET_PATH" \
  --checkpoint "$CHECKPOINT" \
  > "$POLYBEAST_LOG" 2>&1

# Kill the background pantheon process.
echo "Done training, killing pantheon."
kill -9 "$BG_PID"
pkill -9 -f "pantheon"

echo "Testing..."
"$ROOT_DIR"/scripts/test.sh \
  --checkpoint "$CHECKPOINT" \
  --max_jobs "$MAX_JOBS" \
  --job_ids "$JOB_IDS" \
  --logdir "$BASE_LOG_DIR"

echo "All done! Model: $CHECKPOINT."
