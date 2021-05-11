#!/bin/bash -eu

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -eu

# This script deletes all Pantheon logs found in the pantheon/tmp folder.
# ONLY USE IT WHEN NO EXPERIMENTS ARE RUNNING!

SCRIPT_DIR=`dirname "$0"`
PANTHEON_DIR=`realpath "$SCRIPT_DIR"/../_build/deps/pantheon`
LOG_DIR="$PANTHEON_DIR"/tmp
DEL_DIR="$LOG_DIR.to_delete"
EMPTY_DIR="$LOG_DIR.empty"

echo "Moving Pantheon logs to: $DEL_DIR"
mv "$LOG_DIR" "$DEL_DIR"

# Re-create the directory so that it exists for future experiments.
mkdir "$LOG_DIR"

echo "Deleting this folder -- this may take several hours, be patient"
rm -rf "$EMPTY_DIR"
mkdir "$EMPTY_DIR"
# Using `rsync` instead of `rm` because it is faster.
time rsync -a --delete "$EMPTY_DIR/" "$DEL_DIR"
rmdir "$DEL_DIR"

echo "Done!"
