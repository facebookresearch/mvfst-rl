#!/bin/bash -eu

set -eu

## Usage: ./setup.sh [--inference]

# Note: Pantheon requires python 2.7 while torchbeast needs python3.7.
# Make sure your default python in conda env in python2.7 with an explicit
# python3 command pointing to python 3.7

# ArgumentParser
INFERENCE=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --inference )
      # If --inference is specified, only get what we need to run inference
      INFERENCE=true
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

BUILD_ARGS=""
if [ "$INFERENCE" = true ]; then
  echo -e "Inference-only build"
  BUILD_ARGS="--inference"
fi


CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

BASE_DIR="$PWD"
BUILD_DIR="$BASE_DIR"/_build
DEPS_DIR="$BUILD_DIR"/deps
mkdir -p "$DEPS_DIR"

PANTHEON_DIR="$DEPS_DIR"/pantheon
LIBTORCH_DIR="$DEPS_DIR"/libtorch
TORCHBEAST_DIR="$BASE_DIR"/third-party/torchbeast
MVFST_DIR="$BASE_DIR"/third-party/mvfst

cd "$BASE_DIR"
git submodule sync && git submodule update --init --recursive

function setup_pantheon() {
  # We clone Pantheon into _build/deps instead of using git submodule
  # to avoid circular dependency - pantheon/third_party/ has
  # this project as a submodule. For now, we clone and symlink
  # pantheon/third_party/mv-rl-fst to $BASE_DIR.
  if [ ! -d "$PANTHEON_DIR" ]; then
    echo -e "Cloning Pantheon into $PANTHEON_DIR"
    # TODO (viswanath): Update repo url
    git clone git@github.com:fairinternal/pantheon.git "$PANTHEON_DIR"
  fi

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
  cp /usr/bin/mm-*  "$CONDA_PREFIX"/bin/
  sudo chown root:root "$CONDA_PREFIX"/bin/mm-*
  sudo chmod 4755 "$CONDA_PREFIX"/bin/mm-*

  # Install pantheon tunnel in the conda env.
  cd third_party/pantheon-tunnel && ./autogen.sh \
  && ./configure --prefix="$CONDA_PREFIX" \
  && make -j && sudo make install

  # Force-symlink pantheon/third_party/mv-rl-fst to $BASE_DIR
  # to avoid double-building
  echo -e "Symlinking $PANTHEON_DIR/third_party/mv-rl-fst to $BASE_DIR"
  rm -rf $PANTHEON_DIR/third_party/mv-rl-fst
  ln -sf "$BASE_DIR" $PANTHEON_DIR/third_party/mv-rl-fst

  echo -e "Done setting up Pantheon"
}

function setup_libtorch() {
  # Install CPU-only build of PyTorch libs so that C++ executables of
  # mv-rl-fst such as traffic_gen don't need to be unnecessarily linked
  # with CUDA libs, especially during inference.
  echo -e "Installing libtorch CPU-only build into $LIBTORCH_DIR"
  cd "$DEPS_DIR"

  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip

  # This creates and populates $LIBTORCH_DIR
  unzip libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
  rm -f libtorch-cxx11-abi-shared-with-deps-1.2.0.zip

  echo -e "Done installing libtorch"
}

function setup_grpc() {
  # Manually install grpc. We need this for mv-rl-fst in training mode.
  # Note that this gets installed within the conda prefix which needs to be
  # exported to cmake.
  echo -e "Installing grpc"
  conda install -y -c anaconda protobuf
  cd "$BASE_DIR" && ./third-party/install_grpc.sh
  echo -e "Done installing grpc"
}

function setup_torchbeast() {
  echo -e "Installing TorchBeast"
  cd "$TORCHBEAST_DIR"

  module load NCCL/2.2.13-1-cuda.9.2

  # TorchBeast requires PyTorch with CUDA. This doesn't conflict the CPU-only
  # libtorch installation as the install locations are different.
  # TODO (viswanath): Update path
  echo -e "Installing PyTorch with CUDA for TorchBeast"
  python3 -m pip install /private/home/thibautlav/wheels/torch-1.1.0-cp37-cp37m-linux_x86_64.whl

  # requirements_polybeast.txt includes nest which requires pybind11.
  python3 -m pip install pybind11

  python3 -m pip install -r requirements_polybeast.txt

  # We don't necessarily need to install libtorchbeast (we can get that from
  # wheel), but setup.py also generates rpcenv protobuf files within
  # torchbeast/libtorchbeast/ which we need.
  # Remove previous installation first to make sure all files are overwritten.
  python3 -m pip uninstall -y libtorchbeast
  export LD_LIBRARY_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib:${LD_LIBRARY_PATH}
  CXX=c++ python3 setup.py install

  echo -e "Done installing TorchBeast"
}

function setup_mvfst() {
  echo -e "Installing mvfst"

  # Build and install mvfst
  cd "$MVFST_DIR" && ./build_helper.sh
  cd _build/build/ && make install

  echo -e "Done installing mvfst"
}

if [ "$INFERENCE" = false ]; then
    setup_pantheon
    setup_grpc
    setup_torchbeast
fi
setup_libtorch
setup_mvfst

echo -e "Building mv-rl-fst"
cd "$BASE_DIR" && ./build.sh $BUILD_ARGS
