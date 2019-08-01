#!/bin/bash

CUR_DIR=$(dirname "$0")
ROOT_DIR="$CUR_DIR"/..
TORCHBEAST_DIR="$ROOT_DIR"/third-party/torchbeast

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

SOCKET_PATH="/tmp/rl_server_path"
rm -f $SOCKET_PATH

# TODO (viswanath): More params, also start pantheon_env.py
PYTHONPATH=$PYTHONPATH OMP_NUM_THREADS=1 python3 $ROOT_DIR/train/polybeast.py \
  --address "unix:$SOCKET_PATH"
