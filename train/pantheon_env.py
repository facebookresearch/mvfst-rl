#!/usr/bin/env python3

import os
from os import path
import argparse
import logging
import subprocess
import random
import shlex
import threading
import time
import utils

from constants import SRC_DIR, PANTHEON_ROOT

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description="Pantheon Environment Instances")

parser.add_argument(
    "--mode", default="train", choices=["train", "test"], help="Training or test mode."
)
parser.add_argument(
    "--num_actors",
    type=int,
    default=0,
    help="Number of parallel actors for training. Default 0 starts one actor process per pantheon job.",
)
parser.add_argument(
    "--max_jobs",
    type=int,
    default=0,
    help="Maximum number of different Pantheon emulated experiments to use (0 for all)",
)
parser.add_argument(
    "--job_ids",
    type=str,
    default="",
    help="Comma separate list of job ids. If set, filter and run only the specified pantheon jobs.",
)
parser.add_argument(
    "--server_address",
    type=str,
    default="unix:/tmp/rl_server_path",
    help="RL server address - <host>:<port> or unix:<path>",
)
parser.add_argument(
    "--test_runs_per_job",
    type=int,
    default=5,
    help="Number of episodes to run per experiment in test mode.",
)
parser.add_argument(
    "--logdir",
    type=str,
    default=path.join(SRC_DIR, "train/logs/pantheon"),
    help="Pantheon logs output directory",
)
parser.add_argument(
    "-v", type=int, default=0, help="Verbose log-level for Pantheon sender"
)


def train_run(flags, jobs, thread_id):
    """
    Each pantheon job runs for a default of 30s (max 60s allowed).
    We treat each such run as a separate episode for training and run
    randomly chosen job in parallel.
    """
    pantheon_env = get_pantheon_env(flags)
    episode = 0
    while True:
        # Pick a random experiment to run
        job_id = random.choice(range(len(jobs)))
        cfg, cmd_tmpl = jobs[job_id]

        # Expand data_dir in cmd template
        data_dir = path.join(
            flags.logdir, "train_tid{}_run{}_expt{}".format(thread_id, episode, job_id)
        )
        cmd = utils.safe_format(cmd_tmpl, {"data_dir": data_dir})
        cmd = update_cmd(cmd, flags)

        logging.info(
            "Thread: {}, episode: {}, experiment: {}, cmd: {}".format(
                thread_id, episode, job_id, " ".join(cmd)
            )
        )
        p = subprocess.Popen(cmd, env=pantheon_env)
        p.wait()
        episode += 1


def test_run(flags, jobs, thread_id):
    """
    Thread i runs jobs[i % len(jobs)] flags.test_runs_per_job times.
    """
    job_id = thread_id % len(jobs)
    cfg, cmd_tmpl = jobs[job_id]
    logging.info("Test run: thread {} -> job {}".format(thread_id, job_id))

    pantheon_env = get_pantheon_env(flags)
    episode = 0
    while episode < flags.test_runs_per_job:
        # Expand data_dir in cmd template
        data_dir = path.join(flags.logdir, "test_expt{}_run{}".format(job_id, episode))
        cmd = utils.safe_format(cmd_tmpl, {"data_dir": data_dir})
        cmd = update_cmd(cmd, flags)

        logging.info(
            "Thread: {}, episode: {}, experiment: {}, cmd: {}".format(
                thread_id, episode, job_id, " ".join(cmd)
            )
        )
        p = subprocess.Popen(cmd, env=pantheon_env)
        p.wait()
        episode += 1


def run_pantheon(flags, jobs, num_threads, run_fn):
    logging.info("Launching {} jobs over {} threads for {}.".format(len(jobs), num_threads, flags.mode))

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=run_fn, args=(flags, jobs, i))
        thread.start()
        threads.append(thread)
        # Stagger the beginning of each thread to avoid some errors due to
        # starting a bunch of Pantheon tunnels at once.
        time.sleep(1)

    for thread in threads:
        thread.join()
    logging.info("Done with {}.".format(flags.mode))


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
            "--cc_env_mode=remote",
            "--cc_env_rpc_address={}".format(flags.server_address),
            "--cc_env_time_window_ms=100",
            "-v={}".format(flags.v),
        ]
    )
    cmd = shlex.split(cmd) + ['--extra_sender_args="{}"'.format(extra_sender_args)]
    return cmd


if __name__ == "__main__":
    flags = parser.parse_args()
    all_jobs = get_pantheon_emulated_jobs(flags)

    if flags.job_ids:
        job_ids = [int(job_id) for job_id in flags.job_ids.split(",")]
        jobs = [all_jobs[job_id] for job_id in job_ids]
        logging.info("Filtered {} jobs corresponding to ids {}.".format(len(jobs), flags.job_ids))
    else:
        jobs = all_jobs
        logging.info("Using all {} jobs.".format(len(jobs)))

    if flags.max_jobs > 0:
        logging.info("Filtering a maximum of {} jobs.".format(flags.max_jobs))
        jobs = jobs[:flags.max_jobs]

    if flags.mode == "train":
        # One thread / pantheon env per actor while training
        num_threads = flags.num_actors if flags.num_actors > 0 else len(jobs)
    else:
        # One thread per job to test
        num_threads = len(jobs)

    run_fn = train_run if flags.mode == "train" else test_run
    run_pantheon(flags, jobs, num_threads, run_fn)
