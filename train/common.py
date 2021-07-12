# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Any, Dict

from train.utils import StrEnum


@dataclass
class ConfigCommon:
    # Mode to run in.
    mode: StrEnum("Mode", "train, test, trace") = "train"
    # Number of parallel actors for training (ignored during testing).
    num_actors: int = 40
    # Number of parallel actors used for evaluation purpose during training (ignored
    # during testing). These actors are *taken from* the `num_actors` pool (they are
    # not added to it), and as a result this number must be less than `num_actors`.
    # The data from evaluation actors is only used to collect performance statistics
    # and *not* to update the model.
    num_actors_eval: int = 0
    # The number of actual training actors is equal to `num_actors - num_actors_eval`.
    num_actors_train: int = "${minus: ${num_actors}, ${num_actors_eval}}"
    # RL server address, can be <host>:<port> or unix:<path>".
    server_address: str = "unix:/tmp/rl_server_path"
    # Pantheon and TorchBeast logs output directory".
    logdir: str = "/tmp/logs"
    # File to write torchscript traced model to (for training) or read from
    # (for local testing).
    traced_model: str = "traced_model.pt"
