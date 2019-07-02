#!/bin/bash -eu

# Usage: ./build.sh

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

echo -e "Building mv-rl-fst"

CONDA_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
CMAKE_PREFIX_PATH="$FOLLY_INSTALL_DIR;$MVFST_INSTALL_DIR;$CONDA_PREFIX_PATH"

cd "$CMAKE_BUILD_DIR" || exit
cmake                                        \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"   \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"      \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo          \
  -DBUILD_TESTS=On                           \
  -DCONDA_PREFIX_PATH="$CONDA_PREFIX_PATH"   \
  ../..
make -j "$nproc"
echo -e "Done building."
