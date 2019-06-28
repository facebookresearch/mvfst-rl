#!/bin/bash -eu

# ArgumentParser
FORCE=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --force )
      FORCE=true
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

BASE_DIR="$PWD"
BUILD_DIR=_build
DEPS_DIR="$BUILD_DIR"/deps
mkdir -p "$DEPS_DIR"

PANTHEON_ROOT="$DEPS_DIR"/pantheon

if [ -d "$PANTHEON_ROOT" ] && [ "$FORCE" = true ]; then
  echo -e "$PANTHEON_ROOT already exists but --force specified. Cleaning up."
  rm -rf $PANTHEON_ROOT
fi

function setup_pantheon() {
  # We clone Pantheon into _build/deps instead of using git submodule
  # to avoid circular dependency - pantheon/third_party/ has
  # this project as a submodule. For now, we clone and symlink
  # pantheon/third_party/mv-rl-fst to $BASE_DIR.
  echo -e "Cloning Pantheon into $PANTHEON_ROOT"
  # TODO (viswanath): Update repo url
  git clone git@github.com:fairinternal/pantheon.git $PANTHEON_ROOT

  echo -e "Setting up Pantheon"
  cd $PANTHEON_ROOT
  ./tools/fetch_submodules.sh
  ./tools/install_deps.sh

  # Force-symlink pantheon/third_party/mv-rl-fst to $BASE_DIR
  # to avoid double-building
  echo -e "Symlinking $PANTHEON_ROOT/third_party/mv-rl-fst to $BASE_DIR"
  rm -rf third_party/mv-rl-fst
  ln -sf "$BASE_DIR" third_party/mv-rl-fst
}

if [ -d "$PANTHEON_ROOT" ]; then
  echo -e "$PANTHEON_ROOT already exists, moving on."
else
  setup_pantheon
fi

echo -e "Building mv-rl-fst"
cd "$BASE_DIR" && ./build.sh
