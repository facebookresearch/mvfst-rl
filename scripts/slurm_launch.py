#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Run as python3 -m scripts.slurm_launch

import argparse
import datetime
import glob
import itertools
import logging
import os
import pickle as pkl
from pprint import pprint

import submitit

from train import train

logging.basicConfig(level=logging.INFO)

os.environ["OMP_NUM_THREADS"] = "1"


SWEEP_GRID = dict(
    num_actors=40,
    unroll_length=80,
    total_steps=1000000,
    learning_rate=0.00001,
    use_lstm=[False, True],
    epsilon=0.01,
    entropy_cost=0.01,
    hidden_size=512,
    num_actions=[5],
    reward_clipping=["none", "soft_asymmetric"],
    cc_env_history_size=[0, 1, 10, 20, 50],
    cc_env_norm_ms=100.0,
    cc_env_norm_bytes=1000.0,
    cc_env_time_window_ms=[100],
    cc_env_reward_throughput_factor=1.0,
    cc_env_reward_delay_factor=[0.1, 0.2, 0.25],
    cc_env_reward_packet_loss_factor=0.0,
    cc_env_reward_max_delay=True,
    loglevel=1,
)


def add_args(parser):
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument(
        "--test_mode", default=None, choices=["local", "remote"], help="Test only mode."
    )
    parser.add_argument("--logdir", type=str, default=None, help="For test only mode.")


# key => k; some_key => sk
def make_prefix(key):
    tokens = key.split("_")
    return "".join(w[0] for w in tokens)


def expand_args(params, runs=1):
    sweep_args = {k: v for k, v in params.items() if isinstance(v, list)}
    # sweep :: [{arg1: val1, arg2: val1}, {arg1: val2, arg2: val2}, ...]
    sweep = [
        dict(zip(sweep_args.keys(), vs))
        for vs in itertools.product(*sweep_args.values())
    ]
    expanded = []
    for swargs in sweep:
        for n in range(runs):
            new_args = {**params, **swargs}  # shallow merge
            new_args["xpid"] = "{}--{:02d}".format(
                "-".join([f"{make_prefix(k)}{v}" for k, v in swargs.items()]), n
            )
            expanded.append(new_args)
    return expanded


# Creating cmd-like args
def make_command(params):
    params = itertools.chain(*[("--%s" % k, str(v)) for k, v in params.items()])
    return list(params)


def get_observation_length(history_size, num_actions):
    # State summary stats (5 * 20) + history_size * (one-hot actions + cwnd)
    return 100 + history_size * (num_actions + 1)


def get_actions(num_actions):
    ACTIONS = {
        5: "0,/2,-10,+10,*2",
        7: "0,/2,/1.5,-10,+10,*1.5,*2",
        9: "0,/2,/1.5,/1.25,-10,+10,*1.25,*1.5,*2",
        11: "0,/5,/2,/1.5,/1.25,-10,+10,*1.25,*1.5,*2,*5",
    }
    assert num_actions in ACTIONS, "Unsupported num_actions"
    return ACTIONS[num_actions]


def get_executor(flags, logdir):
    if flags.local:
        executor = submitit.LocalExecutor(folder=logdir)
    else:
        executor = submitit.SlurmExecutor(folder=logdir)

    if flags.test_mode is None:
        executor.update_parameters(
            partition="learnfair",
            time=600,
            nodes=1,
            ntasks_per_node=1,
            job_name="mvfstrl",
            num_gpus=2,
            cpus_per_task=80,
            mem="64GB",
            constraint="pascal",
        )
    else:
        executor.update_parameters(
            partition="learnfair",
            time=120,
            nodes=1,
            ntasks_per_node=1,
            job_name="mvfstrl",
            num_gpus=0,
            cpus_per_task=80,
            mem="64GB",
        )

    return executor


def launch_train(flags):
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    sweep_grid = expand_args(SWEEP_GRID)
    logging.info("Sweeping over {} settings".format(len(sweep_grid)))

    for i, train_args in enumerate(sweep_grid):
        uid = "{}-{}".format(now, train_args["xpid"])
        logdir = "/checkpoint/{}/mvfstrl/{}".format(os.environ["USER"], uid)
        os.makedirs(logdir, exist_ok=True)

        train_args.update(
            {
                "base_logdir": logdir,
                "observation_length": get_observation_length(
                    train_args["cc_env_history_size"], train_args["num_actions"]
                ),
                "cc_env_actions": get_actions(train_args["num_actions"]),
            }
        )

        train_parser = train.get_parser()
        train_flags = train_parser.parse_args(make_command(train_args))

        executor = get_executor(flags, logdir)
        job = executor.submit(train.main, train_flags)
        logging.info(
            "Submitted train job {}/{}, id: {}, logdir: {}:".format(
                i + 1, len(sweep_grid), job.job_id, logdir
            )
        )
        pprint(train_args)


def launch_test(flags):
    assert flags.test_mode is not None
    assert flags.logdir and os.path.exists(
        flags.logdir
    ), "--logdir must be specified and should exist for test only mode"

    submitit_files = glob.glob(os.path.join(flags.logdir, "*_submitted.pkl"))
    assert len(submitit_files) > 0, "Couldn't find submitit submission pkl file"
    with open(submitit_files[0], "rb") as f:
        obj = pkl.load(f)
        test_flags = obj.args[0]
        test_flags.mode = "test_local" if flags.test_mode == "local" else "test"

    executor = get_executor(flags, flags.logdir)
    job = executor.submit(train.main, test_flags)
    logging.info(
        "Submitted test job, id: {}, logdir: {}:".format(job.job_id, flags.logdir)
    )
    pprint(test_flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    flags = parser.parse_args()

    if flags.test_mode is None:
        launch_train(flags)
    else:
        launch_test(flags)
