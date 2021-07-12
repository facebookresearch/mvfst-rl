#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from copy import deepcopy
from pathlib import Path

from omegaconf.omegaconf import DictConfig, OmegaConf
from train import resolvers

from gym_env.env import MvFstEnv
from gym_env.mtenv import MultiEnvWrapper
from mtenv import MTEnv


def get_jobs_to_perform(flags, job_ids):
    jobs = list(flags.env_configs.jobs.values())
    if job_ids:
        jobs = [jobs[job_id] for job_id in job_ids]
    if flags.max_jobs > 0 and flags.max_jobs < len(jobs):
        jobs = jobs[0 : flags.max_jobs]

    return jobs


def build_env(flags: DictConfig, env_index: int, env_config_index: int) -> MvFstEnv:
    # There are cases where the env config index!=env index. For example,
    # we may want to make multiple environments with the same config.
    # env_config_index corresponds to the job_index in rest of the code.
    config = deepcopy(flags)

    address = f"/tmp/rl_server_path_{env_index}"
    try:
        os.remove(address)
    except OSError:
        pass
    config.server_address = "unix:{}".format(address)
    server_cfg = OmegaConf.create(
        {
            "address": config.server_address,
            "max_parallel_calls": 10,
        }
    )

    # env_configs = list(flags.train_jobs.values())

    env_configs = get_jobs_to_perform(flags, job_ids=flags.env_ids)

    resolvers.seed_thread_rng(flags.jobs_seed + env_index)

    from train import pantheon_env as pantheon_env_utils  # type: ignore[import]
    from train.utils import get_observation_length  # type: ignore[import]

    pantheon_env_variables = pantheon_env_utils.get_pantheon_env(
        config, actor_id=env_index
    )

    data_dir = Path(config.logdir) / f"train_tid{env_index}_expt{env_config_index}"

    _, command = pantheon_env_utils.get_job_and_cmd(
        config,
        jobs=env_configs,
        job_id=env_config_index,
        actor_id=env_index,
        data_dir=data_dir,
        job_count=-1,
        job_info_queue=None,
    )

    pantheon_cfg = OmegaConf.create(
        {
            "process_timeout": 2,
            "silence_logs": True,
        }
    )

    observation_length = get_observation_length(
        history_size=flags.cc_env_history_size, num_actions=config.num_actions
    )

    env = MvFstEnv(
        env_id=str(env_index),
        num_actions=config.num_actions,
        observation_length=observation_length,
        server_cfg=server_cfg,
        pantheon_cfg=pantheon_cfg,
        pantheon_command=command,
        pantheon_env_variables=pantheon_env_variables,
    )

    return env


def build_mtenv(flags: DictConfig) -> MTEnv:

    env_config_indices = flags.env_ids
    funcs_to_make_envs = []

    def _func_to_make_one_env(env_index: int, env_config_index: int):
        def _func():
            return build_env(
                flags=flags, env_index=env_index, env_config_index=env_config_index
            )

        return _func

    for env_index, env_config_index in enumerate(env_config_indices):
        funcs_to_make_envs.append(
            _func_to_make_one_env(
                env_index=env_index, env_config_index=env_config_index
            )
        )

    initial_task_state = 0

    env = MultiEnvWrapper(
        funcs_to_make_envs=funcs_to_make_envs,
        initial_task_state=initial_task_state,
    )

    return env
