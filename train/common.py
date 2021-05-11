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
class ConfigCommon(Dict[str, Any]):
    # Mode to run in.
    mode: StrEnum("Mode", "train, test, trace") = "train"
    # Number of parallel actors for training (ignored during testing).
    num_actors: int = 40
    # RL server address, can be <host>:<port> or unix:<path>".
    server_address: str = "unix:/tmp/rl_server_path"
    # Pantheon and TorchBeast logs output directory".
    logdir: str = "/tmp/logs"
    # File to write torchscript traced model to (for training) or read from
    # (for local testing).
    traced_model: str = "traced_model.pt"
