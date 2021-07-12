#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import hydra
import torch
import torch.multiprocessing as mp
from train import train as setup_utils  # type: ignore[import]
from train import utils
from train.constants import CONF_ROOT, THIRD_PARTY_ROOT  # type: ignore[import]

utils.add_to_path(THIRD_PARTY_ROOT)


from gym_env import builder as env_builder


def play_with_single_env(flags):
    env_index, env_config_index = 0, 0
    env = env_builder.build_env(
        flags, env_index=env_index, env_config_index=env_config_index
    )
    env.reset()
    action = torch.tensor([env.action_space.sample()], dtype=torch.int64)
    for step in range(100000):
        obs, reward, done, info = env.step(action)
        action = torch.tensor([env.action_space.sample()], dtype=torch.int64)
        if done:
            env.reset()
            print("resetting the env")
        print(f"step: {step}, reward: {reward}, done: {done}")


def play_with_with_multiple_envs(flags):
    env = env_builder.build_mtenv(flags)
    env.seed_task(0)
    env.reset()
    action = torch.tensor([env.action_space.sample()], dtype=torch.int64)
    for step in range(100000):
        obs, reward, done, info = env.step(action)
        action = torch.tensor([env.action_space.sample()], dtype=torch.int64)
        if done:
            print("resetting the env and the task")
            env.reset_task_state()
            env.reset()

        print(f"obs: {obs}, step: {step}, reward: {reward}, done: {done}")


def play(flags):

    flags.cc_env_mode = "remote"
    setup_utils.init_logdirs(flags)

    env_config_indices = flags.env_ids
    if len(env_config_indices) == 1:
        play_with_single_env(flags)
    else:
        play_with_with_multiple_envs(flags)


@hydra.main(config_path=CONF_ROOT, config_name="config")
def main(flags):
    setup_utils.init(flags)
    play(flags)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
