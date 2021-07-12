# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from os import path


# Paths.
SRC_DIR = path.abspath(path.join(path.dirname(__file__), os.pardir))
PANTHEON_ROOT = path.join(SRC_DIR, "_build", "deps", "pantheon")
CONF_ROOT = path.join(SRC_DIR, "config")
EXPERIMENTS_CFG = path.join(SRC_DIR, "train", "experiments.yml")
THIRD_PARTY_ROOT = path.join(SRC_DIR, "third-party")
TORCHBEAST_ROOT = path.join(SRC_DIR, "third-party", "torchbeast")
GALA_ROOT = path.join(SRC_DIR, "third-party", "gala")
TRACES_ROOT = path.join(SRC_DIR, "train", "traces")


# Numbers.
MAX_CWND = 2000  # should match kDefaultMaxCwndInMss in QuicConstants.h
MM_PACKET_SIZE = 1500  # packet size used by mm-link
UDP_SEND_PACKET_LEN = 1252  # should match QuicConstants.h

# Conversion.
MBYTES_TO_BYTES = 1_000_000