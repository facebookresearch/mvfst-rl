#!/bin/bash -eu

# Usage: ./build.sh [--dbg]

# ArgumentParser
BUILD_TYPE=RelWithDebInfo
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dbg )
      BUILD_TYPE=Debug
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

BASE_DIR="$PWD"
MVFST_DIR="$BASE_DIR"/third-party/mvfst
FOLLY_INSTALL_DIR="$MVFST_DIR"/_build/deps
MVFST_INSTALL_DIR="$MVFST_DIR"/_build

BUILD_DIR=_build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit
BWD=$(pwd)

CMAKE_BUILD_DIR="$BWD"/build
mkdir -p "$CMAKE_BUILD_DIR"
INSTALL_DIR="$BWD"

# Default to parallel build width of 4.
# If we have "nproc", use that to get a better value.
set +x
nproc=4
if [ -z "$(hash nproc 2>&1)" ]; then
    nproc=$(nproc)
fi
set -x

# Load CUDA related modules for linking Torch
module unload cuda
module unload cudnn
module unload NCCL
module load cuda/9.2
module load cudnn/v7.3-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2
export CUDA_HOME="/public/apps/cuda/9.2"
export CUDNN_INCLUDE_DIR="/public/apps/cudnn/v7.3/cuda/include"
export CUDNN_LIB_DIR="/public/apps/cudnn/v7.3/cuda/lib64"

CONDA_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo -e "CONDA_PREFIX_PATH: $CONDA_PREFIX_PATH"
echo -e "PYTHON_SITE_PACKAGES: $PYTHON_SITE_PACKAGES"

CMAKE_PREFIX_PATH="$FOLLY_INSTALL_DIR;$MVFST_INSTALL_DIR;$CONDA_PREFIX_PATH;$PYTHON_SITE_PACKAGES"

echo -e "Building mv-rl-fst"
cd "$CMAKE_BUILD_DIR" || exit
cmake                                        \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"   \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"      \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"          \
  -DBUILD_TESTS=On                           \
  -DCONDA_PREFIX_PATH="$CONDA_PREFIX_PATH"   \
  ../..
make -j "$nproc"
echo -e "Done building."
