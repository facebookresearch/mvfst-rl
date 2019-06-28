#!/bin/bash -eu

# ArgumentParser
SKIP_DEPS=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --skip-deps )
      SKIP_DEPS=true
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

BASE_DIR="$PWD"
MVFST_DIR="$PWD"/third-party/mvfst

if [ "$SKIP_DEPS" = false ]; then
  echo -e "Installing mvfst"

  git submodule update --init --recursive

  # Build and install mvfst
  cd "$MVFST_DIR" && ./build_helper.sh
  cd _build/build/ && make install
  cd "$BASE_DIR"
fi

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

cd "$CMAKE_BUILD_DIR" || exit
cmake                                       \
  -DCMAKE_PREFIX_PATH="$FOLLY_INSTALL_DIR;$MVFST_INSTALL_DIR" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"      \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo          \
  -DBUILD_TESTS=On                           \
  ../..
make -j "$nproc"
echo -e "Done building."
