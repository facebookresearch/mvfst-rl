# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from os import path

SRC_DIR = path.abspath(path.join(path.dirname(__file__), os.pardir))
PANTHEON_ROOT = path.join(SRC_DIR, "_build/deps/pantheon")
EXPERIMENTS_CFG = path.join(SRC_DIR, "train/experiments.yml")
THIRD_PARTY_ROOT = path.join(SRC_DIR, "third-party")
TORCHBEAST_ROOT = path.join(SRC_DIR, "third-party/torchbeast")
GALA_ROOT = path.join(SRC_DIR, "third-party/gala")
