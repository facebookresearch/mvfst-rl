#!/bin/bash -eu

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -eu

## Usage: ./setup.sh [--inference] [--skip-mvfst-deps]

# Note: Pantheon requires python 2.7 while torchbeast needs python3.7.
# Make sure your default python in conda env in python2.7 with an explicit
# python3 command pointing to python 3.7

# ArgumentParser
INFERENCE=false
SKIP_MVFST_DEPS=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --inference )
      # If --inference is specified, only get what we need to run inference
      INFERENCE=true
      shift;;
    --skip-mvfst-deps )
      # If --skip-mvfst-deps is specified, don't get mvfst's dependencies.
      SKIP_MVFST_DEPS=true
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

BUILD_ARGS=""
MVFST_ARGS=""
if [ "$INFERENCE" = true ]; then
  echo -e "Inference-only build"
  BUILD_ARGS="--inference"
else
  echo -e "Installing for training"
fi
if [ "$SKIP_MVFST_DEPS" = true ]; then
  echo -e "Skipping dependencies of mvfst"
  MVFST_ARGS="-s"
fi

PREFIX=${CONDA_PREFIX:-"/usr/local"}

BASE_DIR="$PWD"
BUILD_DIR="$BASE_DIR"/_build
DEPS_DIR="$BUILD_DIR"/deps
mkdir -p "$DEPS_DIR"

if [ -d "$BUILD_DIR/build" ]; then
  echo -e "mvfst-rl already installed, skipping"
  exit 0
fi

PANTHEON_DIR="$DEPS_DIR"/pantheon
LIBTORCH_DIR="$DEPS_DIR"/libtorch
PYTORCH_DIR="$DEPS_DIR"/pytorch
THIRDPARTY_DIR="$BASE_DIR"/third-party
TORCHBEAST_DIR="$THIRDPARTY_DIR"/torchbeast
MVFST_DIR="$THIRDPARTY_DIR"/mvfst

cd "$BASE_DIR"
git submodule sync && git submodule update --init --recursive

function setup_pantheon() {
  if [ -d "$PANTHEON_DIR" ]; then
    echo -e "$PANTHEON_DIR already exists, skipping."
    return
  fi

  # We clone Pantheon into _build/deps instead of using git submodule
  # to avoid circular dependency - pantheon/third_party/ has
  # this project as a submodule. For now, we clone and symlink
  # pantheon/third_party/mvfst-rl to $BASE_DIR.
  echo -e "Cloning Pantheon into $PANTHEON_DIR"
  # TODO: Update repo url
  git clone git@github.com:viswanathgs/pantheon.git "$PANTHEON_DIR"

  echo -e "Installing Pantheon dependencies"
  cd "$PANTHEON_DIR"
  ./tools/fetch_submodules.sh

  # Install pantheon deps. Copied from pantheon/tools/install_deps.sh
  # and modified to explicitly use python2-pip and install location.
  sudo apt-get -y install mahimahi ntp ntpdate texlive python-pip
  sudo apt-get -y install debhelper autotools-dev dh-autoreconf iptables \
                          pkg-config iproute2
  sudo python2 -m pip install matplotlib numpy tabulate pyyaml

  # Copy mahimahi binaries to conda env (to be able to run in cluster)
  # with setuid bit.
  cp /usr/bin/mm-* "$PREFIX"/bin/
  sudo chown root:root "$PREFIX"/bin/mm-*
  sudo chmod 4755 "$PREFIX"/bin/mm-*

  # Install pantheon tunnel in the conda env.
  cd third_party/pantheon-tunnel && ./autogen.sh \
  && ./configure --prefix="$PREFIX" \
  && make -j && sudo make install

  # Force-symlink pantheon/third_party/mvfst-rl to $BASE_DIR
  # to avoid double-building
  echo -e "Symlinking $PANTHEON_DIR/third_party/mvfst-rl to $BASE_DIR"
  rm -rf $PANTHEON_DIR/third_party/mvfst-rl
  ln -sf "$BASE_DIR" $PANTHEON_DIR/third_party/mvfst-rl
  echo -e "Done setting up Pantheon"
}

function setup_libtorch() {
  if [ -d "$LIBTORCH_DIR" ]; then
    echo -e "$LIBTORCH_DIR already exists, skipping."
    return
  fi

  # Install CPU-only build of PyTorch libs so that C++ executables of
  # mvfst-rl such as traffic_gen don't need to be unnecessarily linked
  # with CUDA libs, especially during inference.
  echo -e "Installing libtorch CPU-only build into $LIBTORCH_DIR"
  cd "$DEPS_DIR"

  wget --no-verbose https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip

  # This creates and populates $LIBTORCH_DIR
  unzip libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
  rm -f libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
  echo -e "Done installing libtorch"
}

function setup_grpc() {
  # Manually install grpc. We need this for mvfst-rl in training mode.
  # Note that this gets installed within the conda prefix which needs to be
  # exported to cmake.
  echo -e "Installing grpc"
  conda install -y -c anaconda protobuf
  cd "$BASE_DIR" && ./third-party/install_grpc.sh
  echo -e "Done installing grpc"
}

function setup_pytorch() {
  if [ -d "$PYTORCH_DIR" ]; then
    echo -e "$PYTORCH_DIR already exists, skipping."
    return
  fi

  # TorchBeast requires PyTorch with CUDA. This doesn't conflict the CPU-only
  # libtorch installation as the install locations are different.
  echo -e "Installing PyTorch with CUDA for TorchBeast"
  conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
  conda install -y -c pytorch magma-cuda92

  echo -e "Cloning PyTorch into $PYTORCH_DIR"
  git clone -b v1.2.0 --recursive https://github.com/pytorch/pytorch "$PYTORCH_DIR"
  cd "$PYTORCH_DIR"

  export CMAKE_PREFIX_PATH=${PREFIX}
  python3 setup.py install
  echo -e "Done installing PyTorch"
}

function setup_torchbeast() {
  echo -e "Installing TorchBeast"
  cd "$TORCHBEAST_DIR"
  python3 -m pip install -r requirements.txt

  # Install nest
  cd nest/ && CXX=c++ python3 -m pip install . -vv && cd ..

  export LD_LIBRARY_PATH=${PREFIX}/lib:${LD_LIBRARY_PATH}
  CXX=c++ python3 setup.py install
  echo -e "Done installing TorchBeast"
}

function setup_mvfst() {
  # Build and install mvfst
  echo -e "Installing mvfst"
  cd "$MVFST_DIR" && ./build_helper.sh "$MVFST_ARGS"
  cd _build/build/ && make install
  echo -e "Done installing mvfst"
}

if [ "$INFERENCE" = false ]; then
    setup_pantheon
    setup_grpc
    setup_pytorch
    setup_torchbeast
fi
setup_libtorch
setup_mvfst

echo -e "Building mvfst-rl"
cd "$BASE_DIR" && ./build.sh $BUILD_ARGS
