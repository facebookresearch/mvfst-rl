#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Gym-compatible wrapper for the Congestion Control."""
import subprocess
from typing import Any, Dict, List, Tuple

import torch
from gym.core import Env
from gym.spaces import Box, Discrete
from omegaconf.dictconfig import DictConfig
from torchbeast import Server
from torchbeast.queue import Empty as EmptyQueueException
from torchbeast.queue import Queue

ObsType = torch.Tensor
RewardType = torch.Tensor
DoneType = torch.Tensor
InfoType = Dict[str, Any]

ActionType = torch.Tensor

PantheonObservationType = Tuple[ObsType, RewardType, DoneType]

PantheonEnvOutputType = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PantheonObservationType,
]


class MvFstEnv(Env):
    def __init__(
        self,
        env_id: str,
        num_actions: int,
        observation_length: int,
        server_cfg: DictConfig,
        pantheon_cfg: DictConfig,
        pantheon_command: str,
        pantheon_env_variables: Dict[str, Any],
    ):

        self.spec = None

        self.action_space = Discrete(num_actions)
        self.observation_space = Box(low=-float("inf"), high=float("inf"), shape=(observation_length,))

        self.id = env_id

        self._command_to_run_pantheon_process = pantheon_command
        self._env_variables_to_run_pantheon_process = pantheon_env_variables

        self._timeout: int = pantheon_cfg.process_timeout

        self._silence_pantheon_logs = pantheon_cfg.silence_logs

        self._previous_obs_from_pantheon_process: PantheonObservationType
        # This variable tracks the last seen observation from the pantheon process

        self._pantheon_observation_queue: Queue[PantheonObservationType] = Queue(
            maxsize=1
        )
        # This is the queue where the pantheon process writes the observation (via the inference server).

        self._action_queue: Queue[ActionType] = Queue(maxsize=1)
        # This is the queue where the actor's action is written and routed to the pantheon process  (via the inference server).

        self._process = self._run_pantheon_process()
        self._server: Server
        self._init_server(server_cfg=server_cfg)

    def _init_server(self, server_cfg: DictConfig):
        """Initialize the server"""

        self._server = Server(
            address=server_cfg.address, max_parallel_calls=server_cfg.max_parallel_calls
        )
        self._server.bind("inference", self._infer_action, batch_size=None)

        self._server.run()

    def __del__(self):
        """Book-keeping to release resources"""

        self._process.terminate()
        self._pantheon_observation_queue.close()
        self._action_queue.close()
        self._server.stop()

    def _run_pantheon_process(self):
        """Run the pantheon process"""

        if self._silence_pantheon_logs:
            stdout = subprocess.DEVNULL
        else:
            stdout = None

        return subprocess.Popen(
            self._command_to_run_pantheon_process,
            env=self._env_variables_to_run_pantheon_process,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )

    def _infer_action(self, *pantheon_env_output: PantheonEnvOutputType) -> ActionType:
        """Write the output (from pantheon process) to the pantheon observation queue
        and read action from the action queue."""

        self._pantheon_observation_queue.put(pantheon_env_output[3])
        action: ActionType = self._action_queue.get()
        return action

    def reset(self) -> ObsType:
        self._process.terminate()
        # Maye it is better to just terminate the process?
        self._process = self._run_pantheon_process()
        obs_from_pantheon_process, done = self._read_from_pantheon_observation_queue()
        # not considering the case where where done is false
        return obs_from_pantheon_process[0]

    def _read_from_pantheon_observation_queue(
        self,
    ) -> Tuple[PantheonObservationType, bool]:
        done = False
        # This variable indicates both: (i) if the current "episode" is over and (ii) if we can break out of the while loop.
        while not done:
            try:
                obs_from_pantheon_process: PantheonObservationType = (
                    self._pantheon_observation_queue.get(timeout=self._timeout)
                )
                self._previous_obs_from_pantheon_process = obs_from_pantheon_process
                break
            except EmptyQueueException:
                if self._process.poll() is not None:
                    done = True
        return (self._previous_obs_from_pantheon_process, done)

    def step(
        self, action: ActionType
    ) -> Tuple[ObsType, RewardType, DoneType, InfoType]:
        self._action_queue.put(action)
        obs_from_pantheon_process, done = self._read_from_pantheon_observation_queue()
        return (
            obs_from_pantheon_process[0],
            obs_from_pantheon_process[1],
            torch.Tensor([done]),
            {},
        )

    def seed(self, seed=None):
        return
