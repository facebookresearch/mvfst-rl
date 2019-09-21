#!/bin/bash -eu

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Usage: ./build.sh [--dbg] [--clean] [--inference]

# ArgumentParser
BUILD_TYPE=RelWithDebInfo
CLEAN=false
INFERENCE=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dbg )
      BUILD_TYPE=Debug
      shift;;
    --clean )
      CLEAN=true
      shift;;
    --inference )
      # If --inference is specified, only build what we need for inference
      INFERENCE=true
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

PREFIX=${CONDA_PREFIX:-"/usr/local"}

BASE_DIR="$PWD"
BUILD_DIR="$BASE_DIR"/_build
DEPS_DIR="$BUILD_DIR"/deps

LIBTORCH_DIR="$DEPS_DIR"/libtorch
MVFST_DIR="$BASE_DIR"/third-party/mvfst
FOLLY_INSTALL_DIR="$MVFST_DIR"/_build/deps
MVFST_INSTALL_DIR="$MVFST_DIR"/_build

INFERENCE_ONLY=OFF
if [ "$INFERENCE" == true ]; then
  echo -e "Inference-only build"
  INFERENCE_ONLY=ON
fi


mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit
BWD=$(pwd)
CMAKE_BUILD_DIR="$BWD"/build
INSTALL_DIR="$BWD"

if [ "$CLEAN" == true ]; then
  echo -e "Performing a clean build: rm -rf $CMAKE_BUILD_DIR"
  rm -rf $CMAKE_BUILD_DIR
fi
mkdir -p "$CMAKE_BUILD_DIR"

# Default to parallel build width of 4.
# If we have "nproc", use that to get a better value.
set +x
nproc=4
if [ -z "$(hash nproc 2>&1)" ]; then
    nproc=$(nproc)
fi
set -x

# PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

CMAKE_PREFIX_PATH="$PREFIX;$FOLLY_INSTALL_DIR;$MVFST_INSTALL_DIR;$LIBTORCH_DIR"
echo -e "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"

echo -e "Building mvfst-rl"
cd "$CMAKE_BUILD_DIR" || exit
cmake                                        \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"   \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"      \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"           \
  -DPREFIX_PATH="$PREFIX"                    \
  -DBUILD_TESTS=On                           \
  -DINFERENCE_ONLY="$INFERENCE_ONLY"         \
  ../..
make -j "$nproc"
echo -e "Done building."
