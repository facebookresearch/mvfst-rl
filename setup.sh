#!/bin/bash -eu

## Usage: ./setup.sh [--force]

# Note: Pantheon requires python 2.7 while torchbeast needs python3.7.
# Make sure your default python in conda env in python2.7 with an explicit
# python3 command pointing to python 3.7

# ArgumentParser
FORCE=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --force )
      # If --force is specified, even if deps have already been setup,
      # it's cleaned up and re-installed. Otherwise, we skip if pantheon dir
      # already exists.
      FORCE=true
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

BASE_DIR="$PWD"
BUILD_DIR="$BASE_DIR"/_build
DEPS_DIR="$BUILD_DIR"/deps
mkdir -p "$DEPS_DIR"

PANTHEON_DIR="$DEPS_DIR"/pantheon
TORCHBEAST_DIR="$BASE_DIR"/third-party/torchbeast
MVFST_DIR="$BASE_DIR"/third-party/mvfst

cd $BASE_DIR
git submodule sync && git submodule update --init --recursive

function setup_pantheon() {
  # We clone Pantheon into _build/deps instead of using git submodule
  # to avoid circular dependency - pantheon/third_party/ has
  # this project as a submodule. For now, we clone and symlink
  # pantheon/third_party/mv-rl-fst to $BASE_DIR.
  if [ ! -d "$PANTHEON_DIR" ]; then
    echo -e "Cloning Pantheon into $PANTHEON_DIR"
    # TODO (viswanath): Update repo url
    git clone git@github.com:fairinternal/pantheon.git $PANTHEON_DIR
  fi

  echo -e "Installing Pantheon dependencies"
  cd $PANTHEON_DIR

  # Install pantheon deps. Copied from pantheon/tools/install_deps.sh
  # and modified to explicitly use python2-pip and install location.
  sudo apt-get -y install mahimahi ntp ntpdate texlive python-pip
  sudo apt-get -y install debhelper autotools-dev dh-autoreconf iptables \
                          pkg-config iproute2
  sudo python2 -m pip install matplotlib numpy tabulate pyyaml

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

setup_torchbeast() {
  echo -e "Installing TorchBeast"
  cd $TORCHBEAST_DIR

  # Install PyTorch from wheel
  # TODO (viswanath): Update wheel location and torchbeast submodule
  python3 -m pip install /private/home/thibautlav/wheels/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
  python3 -m pip install -r requirements.txt

  # Manually install grpc. We need this for mv-rl-fst.
  # Note that this gets installed within the conda prefix which needs to be
  # exported to cmake.
  conda install -y -c anaconda protobuf python=3.7
  ./install_grpc.sh

  # We don't necessarily need to install libtorchbeast (we can get that from
  # wheel), but setup.py also generates rpcenv protobuf files within
  # torchbeast/libtorchbeast/ which we need.
  export LD_LIBRARY_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib:${LD_LIBRARY_PATH}
  module load NCCL/2.2.13-1-cuda.9.2
  python3 setup.py build develop

  echo -e "Done installing TorchBeast"
}

setup_mvfst() {
  echo -e "Installing mvfst"

  # Build and install mvfst
  cd "$MVFST_DIR" && ./build_helper.sh
  cd _build/build/ && make install

  echo -e "Done installing mvfst"
}

if [ ! -d "$PANTHEON_DIR" ] || [ "$FORCE" = true ]; then
  setup_pantheon
  setup_torchbeast
  setup_mvfst
else
  echo -e "$PANTHEON_DIR already exists, moving on"
fi

echo -e "Building mv-rl-fst"
cd "$BASE_DIR" && ./build.sh
