#!/usr/bin/env python3

import os
from os import path
import argparse
import logging
import subprocess
import random
import utils
import shlex

from constants import SRC_DIR, PANTHEON_ROOT

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description="Pantheon Environment Instances")

parser.add_argument(
    "-N",
    "--num_env",
    type=int,
    default=4,
    help="Number of Pantheon environment instances. "
    "This corresponds to number of actors for RL training.",
)
parser.add_argument(
    "--server_address",
    type=str,
    default="unix:/tmp/rl_server_path",
    help="RL server address - <host>:<port> or unix:<path>",
)
parser.add_argument(
    "--logdir",
    type=str,
    default=path.join(SRC_DIR, "train/logs"),
    help="Pantheon logs output directory",
)

src_path = path.join(PANTHEON_ROOT, "src/experiments/test.py")


def run_pantheon(flags):
    # Each pantheon instance runs for a default of 30s (max 60s allowed).
    # We treat each such run as a separate episode for training and run randomly
    # chosen pantheon experiments in parallel.
    logging.info("Starting {} Pantheon env instances at a time".format(flags.num_env))

    jobs = get_pantheon_emulated_jobs(flags)
    pantheon_env = get_pantheon_env(flags)
    episode_count = 0
    while True:
        processes = []
        for i in range(flags.num_env):
            cfg, cmd = random.choice(jobs)  # Pick a random experiment
            cmd = update_cmd(cmd, flags)
            logging.debug("Launch cmd: {}".format(" ".join(cmd)))
            p = subprocess.Popen(cmd, env=pantheon_env)
            processes.append(p)

        for p in processes:
            p.wait()

        episode_count += flags.num_env
        logging.info("Episode count: {}".format(episode_count))


def get_pantheon_emulated_jobs(flags):
    cfg = utils.expt_cfg["emu"]
    matrix = utils.expand_matrix(cfg["matrix"])

    jobs = []
    for mat_dict in matrix:
        for job_cfg in cfg["jobs"]:
            cmd_tmpl = job_cfg["command"]

            # 1. Expand macros
            cmd_tmpl = utils.safe_format(cmd_tmpl, cfg["macros"])
            # 2. Expand variables in mat_dict
            cmd_tmpl = utils.safe_format(cmd_tmpl, mat_dict)
            # 3. Expand meta
            cmd_tmpl = utils.safe_format(cmd_tmpl, utils.meta)

            data_dir = path.join(flags.logdir, "sc_%d" % job_cfg["scenario"])
            cmd_tmpl = utils.safe_format(cmd_tmpl, {"data_dir": data_dir})

            jobs.append((job_cfg, cmd_tmpl))

    return jobs


def get_pantheon_env(flags):
    # $PATH override to put python2 first for Pantheon
    result = subprocess.run(
        ["dirname $(which python2)"], shell=True, stdout=subprocess.PIPE
    )
    python2_path = result.stdout.decode("utf-8").strip()
    logging.info("Located python2 in {}".format(python2_path))

    pantheon_env = os.environ.copy()
    pantheon_env["PATH"] = ":".join([python2_path, pantheon_env["PATH"]])
    return pantheon_env


def update_cmd(cmd, flags):
    extra_sender_args = " ".join(
        [
            "--cc_env_mode=train",
            "--cc_env_rpc_address={}".format(flags.server_address),
            # TODO (viswanath): Change agg type
            "--cc_env_agg=fixed",
            "--cc_env_fixed_window_size=20",
        ]
    )
    cmd = shlex.split(cmd) + ['--extra_sender_args="{}"'.format(extra_sender_args)]
    return cmd


if __name__ == "__main__":
    flags = parser.parse_args()
    run_pantheon(flags)
