#!/bin/bash -eu

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -eu

## Usage: ./setup.sh [--clean] [--inference] [--skip-mvfst-deps] [--skip-mm-binaries] [--skip-pantheon-deps]

# ArgumentParser
CLEAN_SETUP=false
INFERENCE=false
SKIP_MVFST_DEPS=false
SKIP_MM_BINARIES=false
SKIP_PANTHEON_DEPS=false
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --clean )
      # If --clean is specified, re-install everything from scratch. Note that
      # re-creating the conda env must still be done manually if desired, with:
      #     conda activate base && conda env remove -n mvfst-rl && conda create -n mvfst-rl python=3.8 -y && conda activate mvfst-rl
      CLEAN_SETUP=true
      shift;;
    --inference )
      # If --inference is specified, only get what we need to run inference
      INFERENCE=true
      shift;;
    --skip-mvfst-deps )
      # If --skip-mvfst-deps is specified, don't get mvfst's dependencies.
      SKIP_MVFST_DEPS=true
      shift;;
    --skip-mm-binaries )
      # If --skip-mm-binaries is specified, don't build the mm-* binaries.
      # Instead, the user is responsible for ensuring these binaries are
      # available in the PATH with the correct (root) permissions.
      SKIP_MM_BINARIES=true
      shift;;
    --skip-pantheon-deps )
      # If --skip-pantheon-deps is specified, don't install / build the
      # Pantheon dependencies for its Congestion Control schemes.
      # `mvfst-rl` will still work, but you may not be able to compare to
      # other CC schemes.
      SKIP_PANTHEON_DEPS=true
      shift;;
    * )    # Unknown option
      echo "Invalid command-line argument: '$1'"
      exit 1
  esac
done

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

PANTHEON_DIR="$DEPS_DIR"/pantheon
LIBTORCH_DIR="$DEPS_DIR"/libtorch
PYTORCH_DIR="$DEPS_DIR"/pytorch
THIRDPARTY_DIR="$BASE_DIR"/third-party
TORCHBEAST_DIR="$THIRDPARTY_DIR"/torchbeast
MVFST_DIR="$THIRDPARTY_DIR"/mvfst

function check_env() {
  if [ "$CONDA_DEFAULT_ENV" = base ]; then
    read -r -p "You are in conda's BASE environment, are you sure this is where you want to install mvfst-rl? [y/N] " USER_CONFIRM
    case "$USER_CONFIRM" in
      [yY][eE][sS]|[yY])
        ;;
      *)
        echo "Aborting"
        exit 1
        ;;
    esac
  fi
}

function check_python2() {
  PYTHON2_PATH="$(which python2)" || (echo "Python 2 not found" && exit 1)
  PYTHON2_FOLDER="$(dirname $(which python2))"
  EXPECTED_PYTHON_PATH="$PYTHON2_FOLDER"/python
  if [ ! -f "$EXPECTED_PYTHON_PATH" ]; then
    echo "Expected to find a 'python' executable in the same folder as 'python2', but '$EXPECTED_PYTHON_PATH' does not exist."
    echo "Maybe you need to run 'sudo apt install python'?"
    exit 1
  fi
  PY_MAJOR=$("$EXPECTED_PYTHON_PATH" -c "import sys; print(sys.version_info.major)")
  if [ $PY_MAJOR != 2 ]; then
      # We need to make sure that this is Python 2, because this executable will be used by Pantheon
      # (see `get_pantheon_env()` in pantheon_env.py).
      echo "Expected '$EXPECTED_PYTHON_PATH' to be a Python 2 executable, but this is Python $PY_MAJOR."
      echo "Please ensure it is pointing to '$PYTHON2_PATH'."
      exit 1
  fi
}

function get_sudo() {
    sudo echo Obtained root security privileges
}

function cleanup_git_locks() {
  # Sometimes there may be leftover git locks => delete them before a clean install.
  find .git -name index.lock -exec rm -f {} \;
}

function cleanup_setup_dirs() {
  TODEL_DIRS=()
  for DEL_DIR in "$BUILD_DIR" "$MVFST_DIR" "$TORCHBEAST_DIR"; do
    # Only delete directories that exist and are non-empty (NB: submodules'
    # directories are empty after initial checkout).
    if [[ -d "$DEL_DIR" && -n $(ls -A $DEL_DIR) ]]; then
      TODEL_DIRS+=("$DEL_DIR")
    fi
  done
  TODEL_FILES=()
  if [ "$SKIP_MM_BINARIES" = false ]; then
      TODEL_FILES=($(find "$PREFIX"/bin -maxdepth 1 -name "mm-*"))
  fi
  if [ ${#TODEL_DIRS[@]} -gt 0 ]; then
    echo "--clean will delete the following folders:"
    for DEL_DIR in ${TODEL_DIRS[@]}; do
      echo "  '$DEL_DIR'"
    done
  fi
  if [ ${#TODEL_FILES[@]} -gt 0 ]; then
    echo "--clean will delete the following files:"
    for DEL_FILE in ${TODEL_FILES[@]}; do
      echo "  '$DEL_FILE'"
    done
  fi
  if [ ${#TODEL_DIRS[@]} -gt 0 ] || [ ${#TODEL_FILES[@]} -gt 0 ]; then
    read -r -p "Are you 100% SURE you want to delete ALL the above? [y/N] " USER_CONFIRM
    case "$USER_CONFIRM" in
      [yY][eE][sS]|[yY])
        for DEL_FILE in ${TODEL_FILES[@]}; do
          echo "Deleting '$DEL_FILE'"
          rm -f "$DEL_FILE"
          if [ -f "$DEL_FILE" ]; then
            echo "Failed to delete '$DEL_FILE'"
            exit 1
          fi
        done
        for DEL_DIR in ${TODEL_DIRS[@]}; do
          echo "Deleting '$DEL_DIR'"
          rm -rf "$DEL_DIR"
          if [ -d "$DEL_DIR" ]; then
            echo "Failed to delete '$DEL_DIR'"
            exit 1
          fi
        done
        ;;
      *)
        echo "Aborting"
        exit 1
        ;;
    esac
  fi
}

function setup_conda_dependencies() {
  HYDRA_COMMIT_ID=f07036a8f3895169e62d89ad653434291b994780
  OMEGACONF_COMMIT_ID=a2e25ddc7aa34bf904ce6f111accc08f50557e58

  # `openjdk` is to get java so as to be able to build the dev version of Hydra.
  conda install -y cloudpickle openjdk psutil pylint pyyaml tensorboard
  conda install -y -c conda-forge hiplot
  # A dev version is required for Hydra, to properly work with list arguments in resolvers.
  pip install git+git://github.com/odelalleau/hydra.git@$HYDRA_COMMIT_ID --upgrade --pre
  pip install hydra-submitit-launcher hydra-joblib-launcher hydra-nevergrad-sweeper --upgrade --pre
  # A dev version is also required for OmegaConf, to fix some thread-safety issues.
  pip uninstall omegaconf --yes
  pip install git+git://github.com/odelalleau/omegaconf.git@$OMEGACONF_COMMIT_ID --upgrade --pre
}

function setup_pantheon() {
  PANTHEON_COMMIT_ID=751a054e2b64349b50f25967d36a030ca8f992f5

  if [ -d "$PANTHEON_DIR" ]; then
    echo -e "$PANTHEON_DIR already exists, skipping."
    return
  fi

  # We clone Pantheon into _build/deps instead of using git submodule
  # for legacy reasons (there used to be a circular dependency otherwise).
  # This may be changed in the future.
  echo -e "Cloning Pantheon into $PANTHEON_DIR"
  git clone -b mvfst-rl git@github.com:odelalleau/pantheon.git "$PANTHEON_DIR"
  cd "$PANTHEON_DIR"
  git -c advice.detachedHead=false checkout "$PANTHEON_COMMIT_ID"

  echo -e "Installing Pantheon dependencies"
  ./tools/fetch_submodules.sh

  # Obtain pip for Python 2
  curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output tmp_get-pip.py
  sudo python2 tmp_get-pip.py
  rm -f tmp_get-pip.py

  # Install pantheon deps. Copied from pantheon/tools/install_deps.sh
  # and modified to explicitly use Python2's pip and install location.
  sudo apt-get -y install mahimahi ntp ntpdate texlive
  sudo apt-get -y install debhelper autotools-dev dh-autoreconf iptables \
                          pkg-config iproute2
  # Disable cache directory because of potential permission issues with sudo.
  sudo python2 -m pip install --no-cache-dir matplotlib numpy psutil pyyaml tabulate

  if [ "$SKIP_MM_BINARIES" = false ]; then
    # Copy mahimahi binaries to conda env (to be able to run in cluster)
    # with setuid bit.
    cp /usr/bin/mm-* "$PREFIX"/bin/
    sudo chown root:root "$PREFIX"/bin/mm-*
    sudo chmod 4755 "$PREFIX"/bin/mm-*

    # Install pantheon tunnel in the conda env.
    cd third_party/pantheon-tunnel && ./autogen.sh \
    && ./configure --prefix="$PREFIX" \
    && make -j && sudo make install
  fi

  # Symlink pantheon/third_party/mvfst-rl to $BASE_DIR.
  echo -e "Symlinking $PANTHEON_DIR/third_party/mvfst-rl to $BASE_DIR"
  ln -sf "$BASE_DIR" $PANTHEON_DIR/third_party/mvfst-rl

  echo -e "Done setting up Pantheon"
}

function setup_pantheon_dependencies() {
    echo "Setting up dependencies for Pantheon's CC schemes"
    # Setup other dependencies so as to be able to run other CC algorithms.
    # We only use schemes enabled in `pantheon_env.get_test_schemes()` by default,
    # and skip the `mvfst_*` schemes since everything should be setup already after
    # this script is done.
    CC_SCHEMES="bbr copa cubic fillp fillp_sheep indigo ledbat pcc pcc_experimental scream sprout taova vegas verus vivace"
    cd "$PANTHEON_DIR"
    python2 ./src/experiments/setup.py --install-deps --schemes "${CC_SCHEMES}"
    python2 ./src/experiments/setup.py --setup --schemes "${CC_SCHEMES}"
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

  wget --no-verbose --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcpu.zip

  # This creates and populates $LIBTORCH_DIR
  unzip libtorch-cxx11-abi-shared-with-deps-1.7.1+cpu.zip
  rm -f libtorch-cxx11-abi-shared-with-deps-1.7.1+cpu.zip
  echo -e "Done installing libtorch"
}

function setup_grpc() {
  # Manually install grpc. We need this for mvfst-rl in training mode.
  echo -e "Installing grpc"
  cd "$TORCHBEAST_DIR" && ./scripts/install_grpc.sh
  echo -e "Done installing grpc"
}

function setup_pytorch() {
  if [ -d "$PYTORCH_DIR" ]; then
    echo -e "$PYTORCH_DIR already exists, skipping."
    return
  fi

  # Check that CUDA is available (i.e., nvcc is installed).
  nvcc --version

  # Extract CUDA version (in format of the form 92 / 100 / 101 for instance)
  CUDA_VERSION=$(nvcc --version | \
                 grep -o "release [0-9]\+\.[0-9]\+," | \
                 sed "s/[release |,|\.]//g")

  # TorchBeast requires PyTorch with CUDA. This doesn't conflict with the
  # CPU-only libtorch installation as the install locations are different.
  echo -e "Installing PyTorch with CUDA (v$CUDA_VERSION) for TorchBeast"
  conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
  conda install -y -c pytorch magma-cuda$CUDA_VERSION

  echo -e "Cloning PyTorch into $PYTORCH_DIR"
  git clone -b v1.7.1 --recursive https://github.com/pytorch/pytorch "$PYTORCH_DIR"
  cd "$PYTORCH_DIR"

  CPF=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
  # Support multiple compute capabilities for better compatibility.
  # See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
  # NB: 6.2 does not seem to be supported by NCCL and is thus omitted in that list.
  CMAKE_PREFIX_PATH="$CPF" TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5" python3 setup.py install
  echo -e "Done installing PyTorch"
}

function setup_torchbeast() {
  echo -e "Installing TorchBeast"
  cd "$TORCHBEAST_DIR"
  python3 -m pip install -r requirements.txt

  # Install nest
  python3 -m pip install third_party/nest

  # Compile and install TorchBeast
  LDP=${PREFIX}/lib:${LD_LIBRARY_PATH}
  LD_LIBRARY_PATH="$LDP" CXX=c++ python3 setup.py build develop  # To get some path magic.
  LD_LIBRARY_PATH="$LDP" CXX=c++ python3 setup.py install
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
  # Get sudo access at beginning of setup to avoid asking for password in the middle.
  get_sudo
fi

if [ "$CLEAN_SETUP" = true ]; then
  cleanup_setup_dirs
  cleanup_git_locks
fi

if [ -d "$BUILD_DIR/build" ]; then
  echo -e "mvfst-rl already installed, skipping (use --clean to re-install everything)"
  exit 0
fi

mkdir -p "$DEPS_DIR"

cd "$BASE_DIR"
git submodule sync && git submodule update --init --recursive

if [ "$INFERENCE" = false ]; then
  check_env
  check_python2
  setup_conda_dependencies
  setup_pantheon
  if [ "$SKIP_PANTHEON_DEPS" = true ]; then
    echo -e "Skipping additional Pantheon dependencies"
  else
    setup_pantheon_dependencies
  fi
  setup_pytorch
  setup_grpc
  setup_torchbeast
fi
setup_libtorch
setup_mvfst

echo -e "Building mvfst-rl"
cd "$BASE_DIR" && ./build.sh $BUILD_ARGS
