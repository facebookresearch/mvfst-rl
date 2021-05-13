#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import logging
import os
import subprocess
import random
import shutil
import threading
import time

from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

from train import resolvers, utils
from train.constants import PANTHEON_ROOT, TRACES_ROOT
from train.env_spec import ConfigEnvSpec, ConfigJobs, get_env_cmd, set_scaled_traces
from train.utils import StrEnum, default_empty_dict, default_empty_list, default_list

logging.basicConfig(level=logging.INFO)


@dataclass
class ConfigEnv:
    # Maximum number of different Pantheon emulated experiments to use (0 for all).
    max_jobs: int = 0
    # Comma separate list of job ids. If set, filter and *train* only on the
    # specified pantheon `train_jobs`.
    train_job_ids: List[int] = default_empty_list()
    # Comma separate list of job ids. If set, filter and *evaluate* only on the
    # specified pantheon `eval_jobs`.
    eval_job_ids: List[int] = default_empty_list()
    # Comma separate list of job ids. If set, filter and *test* only on the
    # specified pantheon `eval_jobs`.
    test_job_ids: List[int] = default_empty_list()
    # Comma separate list of congestion control schemes. If set, filter and test
    # only on the specified schemes.
    test_schemes: str = ""
    # Number of runs per job to average results over in test mode.
    test_runs_per_job: int = 3
    # Verbose log-level for Pantheon sender.
    loglevel: int = 1
    # Command line parameters common to all environments.
    common_params: List[str] = default_list(
        ["local", "--data-dir", "{data_dir}", "--pkill-cleanup"]
    )
    # Path to the script launching the environment.
    test_path: str = os.path.join(PANTHEON_ROOT, "src", "experiments", "test.py")
    # Path to the script analyzing results.
    analyze_path: str = os.path.join(PANTHEON_ROOT, "src", "analysis", "analyze.py")
    # Directory containing trace files.
    traces_dir: str = TRACES_ROOT
    # Minimum size (in number of rows) of generated scaled trace files.
    # Larger values lead to more accurate traces but add processing time.
    min_scaled_trace_size: int = 100000
    # Specifications of training jobs.
    train_jobs: ConfigJobs = ConfigJobs()
    # Specifications of evaluation jobs (executed by `num_actors_eval` actors).
    eval_jobs: ConfigJobs = ConfigJobs()
    # Seed used to generate random jobs. Note that using a different number of
    # actors will also change the generated jobs (each actor generates its own
    # jobs, from a unique seed derived from `jobs_seed` and the actor ID).
    jobs_seed: int = 123
    # CongestionControlEnvConfig::Mode. Local only support testing.
    cc_env_mode: StrEnum("CCEnvMode", "local, remote") = "remote"
    # tate aggregation type.
    cc_env_agg: StrEnum("CCEnvAgg", "time, fixed") = "time"
    # Window duration for time window aggregation.
    cc_env_time_window_ms: int = 100
    # Window size for fixed window aggregation.
    cc_env_fixed_window_size: int = 10
    # Whether to use state summary instead of raw states in observation
    # (auto-enabled for time window aggregation).
    cc_env_use_state_summary: bool = True
    # Length of history (such as past actions) to include in observation.
    cc_env_history_size: int = 20
    # Norm factor for temporal fields.
    cc_env_norm_ms: float = 100
    # Norm factor for byte fields.
    cc_env_norm_bytes: float = 1000
    # List of actions specifying how cwnd should be updated.
    # First action should be 0 (no-op).
    # This list is optional: `utils.get_actions()` will be invoked if missing.
    cc_env_actions: List[str] = default_empty_list()
    # If `True`, then instead of
    #   a * throughput - b * delay - c * loss
    # we use as reward
    #   a * log(a' + throughput) - b * log(b' + delay) - c * log(c' + loss)
    cc_env_reward_log_ratio: bool = True
    # Throughput multiplier in reward (a).
    cc_env_reward_throughput_factor: float = 1
    # Offset to add to throughput in log version (a').
    cc_env_reward_throughput_log_offset: float = 1e-5
    # Delay multiplier in reward (b).
    cc_env_reward_delay_factor: float = 0.2
    # Offset to add to delay in log version (b').
    cc_env_reward_delay_log_offset: float = 1e-5
    # Packet loss multiplier in reward (c).
    cc_env_reward_packet_loss_factor: float = 0
    # Offset to add to packet loss in log version (c').
    cc_env_reward_packet_loss_log_offset: float = 1e-5
    # Whether to take max delay over observations in reward (avg otherwise).
    cc_env_reward_max_delay: bool = True
    # Target fixed cwnd value (only used in 'fixed' env mode).
    cc_env_fixed_cwnd: int = 10
    # Window length (in us) of min RTT filter used to estimate delay.
    # Default is to set it to a very large value (here, 10K seconds) to be sure
    # that we do not under-estimate the delay within an episode.
    cc_env_min_rtt_window_length_us: int = 10_000_000_000


def get_actor_data(flags, actor_id):
    """Return the actor's list of jobs and Pantheon environment"""
    # Fetch list of jobs for the current thread.
    jobs = utils.get_jobs(flags, actor_id=actor_id)

    # Each actor (= thread) gets its own seed to sample jobs' parameters.
    resolvers.seed_thread_rng(flags.jobs_seed + actor_id)

    # Environment variables for Pantheon.
    pantheon_env = get_pantheon_env(flags)

    return jobs, pantheon_env


def get_job_cmd(flags, jobs, job_id, actor_id, data_dir):
    """Return the command line associate to a given job in the list of all jobs"""
    # Select the desired job and resolve its parameters (=> sampling random numbers).
    job = copy.deepcopy(jobs[job_id])
    OmegaConf.resolve(job)

    # Generate the uplink & donwlink scaled traces (if needed).
    set_scaled_traces(job, data_dir)

    # Generate the command line (expanding the data directory in the template).
    cmd_tmpl = get_env_cmd(flags, job)
    cmd = [utils.safe_format(c, {"data_dir": data_dir}) for c in cmd_tmpl]
    cmd = update_cmd(cmd, flags, actor_id, job_id)

    return cmd


def train_run(flags, thread_id):
    """
    Each pantheon job runs for a default of 30s (max 60s allowed).
    We treat each such run as a separate episode for training and run
    randomly chosen job in parallel.
    """
    # Fetch list of jobs for the current thread.
    jobs, pantheon_env = get_actor_data(flags, actor_id=thread_id)
    logging.info(f"Thread {thread_id}: running {len(jobs)} jobs")
    # Generator giving the next random job to train on.
    job_ids = utils.balanced_randints(resolvers.thread_data.rng, len(jobs))

    episode = 0
    while True:
        # Pick a random job to run.
        job_id = next(job_ids)
        data_dir = (
            Path(flags.logdir) / f"train_tid{thread_id}_run{episode}_expt{job_id}"
        )

        cmd = get_job_cmd(
            flags, jobs=jobs, job_id=job_id, actor_id=thread_id, data_dir=data_dir
        )

        # Launch the job.
        p = subprocess.Popen(cmd, env=pantheon_env)
        logging.info(
            "Thread: {}, episode: {}, experiment: {}, subprocess: {}, cmd: {}".format(
                thread_id, episode, job_id, p.pid, " ".join(cmd)
            )
        )
        p.wait()
        episode += 1

        log_func = logging.debug if p.returncode == 0 else logging.warning
        log_func(
            "Thread: %s => process %s exited with return code %s",
            thread_id,
            p.pid,
            p.returncode,
        )

        # Remove pantheon logs to free up space (may need to retry in case
        # process has not yet fully exited).
        utils.delete_dir(data_dir, max_tries=3, sleep_time=2)


def test_run(flags, thread_id):
    """
    Thread i runs jobs[i % len(jobs)] flags.test_runs_per_job times.
    """
    jobs, pantheon_env = get_actor_data(flags, actor_id=thread_id)

    job_id = thread_id % len(jobs)
    data_dir = path.join(flags.logdir, "test_expt{}".format(job_id))

    cmd = get_job_cmd(
        flags, jobs=jobs, job_id=job_id, actor_id=thread_id, data_dir=data_dir
    )

    # Run tests
    logging.info(
        "Test run: thread {} -> job {}, cmd: {}".format(
            thread_id, job_id, " ".join(cmd)
        )
    )
    p = subprocess.Popen(cmd, env=pantheon_env)
    p.wait()
    assert p.returncode == 0, "Pantheon script exited with error code {}".format(
        p.returncode
    )

    # Run analysis
    analysis_cmd = [flags.analyze_path, "--data-dir={}".format(data_dir)]
    logging.info(
        "Thread {}, job {}: Running analysis on {}, cmd: {}".format(
            thread_id, job_id, data_dir, " ".join(analysis_cmd)
        )
    )
    p = subprocess.Popen(analysis_cmd, env=pantheon_env)
    p.wait()

    shutil.copyfile(
        path.join(data_dir, "pantheon_summary_mean.pdf"),
        path.join(flags.logdir, "test_expt{}.pdf".format(job_id)),
    )
    logging.info(
        "Test run finished for thread {}, job {}. Results in {}.".format(
            thread_id, job_id, data_dir
        )
    )


def run_pantheon(flags, num_threads, run_fn):
    logging.info("Launching {} threads for {}.".format(num_threads, flags.mode))

    threads = []
    for actor_id in range(num_threads):
        # An independent copy of the config is used in each thread. This is a safeguard
        # against potential thread-safety issues in OmegaConf.
        thread = threading.Thread(target=run_fn, args=(copy.deepcopy(flags), actor_id))
        thread.start()
        threads.append(thread)
        # Stagger the beginning of each thread to avoid some errors due to
        # starting a bunch of Pantheon tunnels at once.
        time.sleep(1)

    for thread in threads:
        thread.join()
    logging.info("Done with {}.".format(flags.mode))


def get_jobs_to_perform(flags, job_ids, mode=None, actor_id=None):
    """Obtain the list of jobs to perform given the current settings"""
    mode = flags.mode if mode is None else mode
    # Filter jobs.
    if mode == "train":
        if actor_id is None or actor_id < flags.num_actors_train:
            jobs = flags.train_jobs.jobs
        else:
            jobs = flags.eval_jobs.jobs
    else:
        jobs = flags.eval_jobs.jobs
    jobs = list(jobs.values())
    if job_ids:
        jobs = [jobs[job_id] for job_id in job_ids]
        logging.info(
            "Filtered {} jobs corresponding to ids {}.".format(len(jobs), job_ids)
        )
    else:
        logging.info("Using all {} jobs.".format(len(jobs)))

    if flags.max_jobs > 0 and flags.max_jobs < len(jobs):
        logging.info("Filtering a maximum of {} jobs.".format(flags.max_jobs))
        jobs = jobs[0 : flags.max_jobs]

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


def get_test_schemes(flags):
    """Return test schemes (as a list) based on current settings"""
    if flags.test_schemes:
        return split_schemes(flags.test_schemes)
    else:  # default list of testing schemes
        return [
            "mvfst_rl",
            "mvfst_rl_fixed",
            "mvfst_rl_random",
            "bbr",
            "copa",
            "cubic",
            "fillp",
            "fillp_sheep",
            "indigo",
            "ledbat",
            # "mvfst_bbr",  # disabled due to falling back on cubic
            "mvfst_copa",
            "mvfst_cubic",
            "mvfst_newreno",
            "pcc",
            "pcc_experimental",
            # "quic",  # disabled due to compilation issues
            "scream",
            "sprout",
            "taova",
            "vegas",
            "verus",
            "vivace",
            # "webrtc",  # disabled due to being quite different
        ]


def split_schemes(schemes):
    """
    Split a comma-separated list of schemes.

    For instance, "bbr,mvfst_rl_fixed{cc_env_fixed_cwnd=2,10}" will become:
        ["bbr", "mvfst_rl_fixed{cc_env_fixed_cwnd=2,10}"]
    """
    split = schemes.split(",")
    # Note that we cannot simply return `split` now, as commas may appear
    # within a given scheme. We need to "re-join" some items in that split. We
    # use the "{" and "}" markers for this purpose.
    opening_pos = -1  # position of most recent "{" seen in `split`
    all_schemes = []
    for i, scheme in enumerate(split):
        if "{" in scheme and "}" not in scheme:
            # First component of a scheme: remember its position.
            assert opening_pos == -1
            opening_pos = i
        elif "}" in scheme and "{" not in scheme:
            # Last component of a scheme: re-join all intermediate components.
            assert opening_pos >= 0
            all_schemes.append(",".join(split[opening_pos : i + 1]))
            opening_pos = -1
        elif opening_pos >= 0:
            # Intermediate component: do nothing until we find the last one.
            pass
        else:
            # That scheme was not split initially, we can simply keep it.
            all_schemes.append(scheme)
    assert opening_pos == -1, schemes  # ensure all { are closed with }
    return all_schemes


def update_cmd(cmd, flags, actor_id, job_id):
    if flags.mode == "train":
        schemes = "mvfst_rl"
        run_times = 1
    else:  # test mode
        schemes = " ".join(get_test_schemes(flags))
        run_times = flags.test_runs_per_job
        # In test mode we ignore the input job ID and force it to -1.
        # This ensures that the model cannot use job-specific information.
        job_id = -1

    extra_sender_args = " ".join(
        [
            "--cc_env_mode={}".format(flags.cc_env_mode),
            "--cc_env_rpc_address={}".format(flags.server_address),
            "--cc_env_actor_id={}".format(actor_id),
            "--cc_env_job_id={}".format(job_id),
            "--cc_env_model_file={}".format(flags.traced_model),
            "--cc_env_agg={}".format(flags.cc_env_agg),
            "--cc_env_time_window_ms={}".format(flags.cc_env_time_window_ms),
            "--cc_env_fixed_window_size={}".format(flags.cc_env_fixed_window_size),
            "--cc_env_use_state_summary={}".format(flags.cc_env_use_state_summary),
            "--cc_env_history_size={}".format(flags.cc_env_history_size),
            "--cc_env_norm_ms={}".format(flags.cc_env_norm_ms),
            "--cc_env_norm_bytes={}".format(flags.cc_env_norm_bytes),
            "--cc_env_actions={}".format(",".join(flags.cc_env_actions)),
            "--cc_env_reward_log_ratio={}".format(flags.cc_env_reward_log_ratio),
            "--cc_env_reward_throughput_factor={}".format(
                flags.cc_env_reward_throughput_factor
            ),
            "--cc_env_reward_throughput_log_offset={}".format(
                flags.cc_env_reward_throughput_log_offset
            ),
            "--cc_env_reward_delay_factor={}".format(flags.cc_env_reward_delay_factor),
            "--cc_env_reward_delay_log_offset={}".format(
                flags.cc_env_reward_delay_log_offset
            ),
            "--cc_env_reward_packet_loss_factor={}".format(
                flags.cc_env_reward_packet_loss_factor
            ),
            "--cc_env_reward_packet_loss_log_offset={}".format(
                flags.cc_env_reward_packet_loss_log_offset
            ),
            "--cc_env_reward_max_delay={}".format(flags.cc_env_reward_max_delay),
            "--cc_env_fixed_cwnd={}".format(flags.cc_env_fixed_cwnd),
            "--cc_env_min_rtt_window_length_us={}".format(
                flags.cc_env_min_rtt_window_length_us
            ),
            "-v={}".format(flags.loglevel),
        ]
    )
    return cmd + [
        "--schemes={}".format(schemes),
        "--run-times={}".format(run_times),
        '--extra-sender-args="{}"'.format(extra_sender_args),
    ]


def main(flags):

    if flags.mode == "train":
        # One thread / pantheon env per actor while training.
        num_threads = flags.num_actors
        run_fn = train_run
    else:
        # One thread per job to test.
        num_threads = utils.get_n_jobs(flags)
        run_fn = test_run

    run_pantheon(flags, num_threads, run_fn)
