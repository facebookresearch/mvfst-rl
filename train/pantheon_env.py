#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import itertools
import logging
import os
import queue
import subprocess
import shutil
import threading
import time

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from train import resolvers, utils
from train.constants import (
    MAX_CWND,
    MBYTES_TO_BYTES,
    MM_PACKET_SIZE,
    PANTHEON_ROOT,
    TRACES_ROOT,
    UDP_SEND_PACKET_LEN,
)
from train.env_spec import ConfigJobs, get_env_cmd, set_scaled_traces
from train.utils import (
    StrEnum,
    default_empty_list,
    default_list,
    get_n_jobs,
    get_slurm_temporary_dir,
    make_one_hot,
)

logging.basicConfig(level=logging.INFO)


# Map a trace to its index (used to define the one-hot representing the trace).
# WARNING: this dictionary should be kept in synch with the list of traces
# that may be randomly sampled (ideally it would be defined dynamically from
# this list).
TRACE_TO_ID = {
    k: i
    for i, k in enumerate(
        [
            "0.57mbps-poisson.trace",
            "2.64mbps-poisson.trace",
            "3.04mbps-poisson.trace",
            "100.42mbps.trace",
            "77.72mbps.trace",
            "114.68mbps.trace",
        ]
    )
}

# Map a trace to the (approximate) max throughput it can sustain, in MBytes/s.
TRACE_TO_MAX_THROUGHPUT = {
    "0.57mbps-poisson.trace": 0.069,
    "2.64mbps-poisson.trace": 0.278,
    "3.04mbps-poisson.trace": 0.330,
    "12mbps.trace": 1.5,  # must be updated empirically
    "77.72mbps.trace": 9.32,
    "96mbps.trace": 12,  # must be updated empirically
    "100.42mbps.trace": 12.06,
    "114.68mbps.trace": 13.77,
}


@dataclass
class ConfigEnv:
    # Maximum number of different Pantheon emulated experiments to use (0 for all).
    max_jobs: int = 0
    # Comma separate list of job ids. If set, filter and *train* only on the
    # specified environment ids.
    env_ids: List[int] = default_empty_list()
    # Specifications of environment configs.
    env_configs: ConfigJobs = ConfigJobs()
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
    # Whether, when sampling random jobs, we allow those whose estimated target
    # optimal CWND is higher than `MAX_CWND`.
    allow_jobs_with_target_cwnd_above_max_cwnd: bool = False
    # Size of the task observation.
    # It is currently hardcoded in the config and must be kept in synch with the
    # implementation of `get_task_obs()`.
    task_obs_size: int = 13
    # If True, then in order to obtain the base RTT for a given job, we run a
    # "dummy" episode with relatively low CWND and measure the observed RTT
    # (instead of relying on the job's delay value).
    # This is done only during training (not during testing).
    empirical_base_rtt: bool = False
    # CongestionControlEnvConfig::Mode. Local only support testing.
    cc_env_mode: StrEnum("CCEnvMode", "local, remote, fixed, random") = "remote"
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
    # Which formula to use for the reward:
    #   - linear: a * throughput - b * delay - c * loss
    #   - log_ratio:
    #       a * log(a' + throughput) - b * log(b' + delay) - c * log(c' + loss)
    #   - min_throughput:
    #       a * rt + b * rd
    #       rt:
    #           1 if throughput >= r * link_bandwdith
    #           1 - 2 / (1 + throughput / link_bandwidth) otherwise (< 0)
    #       rd:
    #           log(k) - log(k + n_packets_in_queue)
    #           n_packets_in_queuex = delay * link_bandwidth / packet_size
    #   - target_cwnd:
    #       a * (1 - min(1, abs(1 - cwnd / target_cwnd)))
    #       - b * (cwnd <= target_cwnd ? 0 : min(1, delay / rtt_min))
    #   - target_cwnd_shaped:
    #       a * (cwnd_is_closer_to_target_cwnd_compared_to_previous_step ? 1 : -1)
    #       (also +1 if cwnd is exactly equal to target_cwnd)
    #   - higher_is_better (debugging):
    #       a * (cwnd_is_higher_than_previous_step_or_above_target_cwnd ? 1 : -1)
    #   - above_cwnd:
    #       a * (cwnd >= r * target_cwnd ? 1 : 0) - b * log(b' + delay)
    #   - cwnd_range
    #       a * (r * target_cwnd <= cwnd <= r' * target_cwnd ? 1 : 0)
    #   - cwnd_range_soft
    #       a * (low <= cwnd <= high ? 1 - 2 * abs(cwnd - mid) / (high - low)  : 0)
    #       low = r * target_cwnd
    #       high = r' * target_cwnd
    #       mid = (low + high) / 2
    #   - cwnd_tradeoff
    #       (a * (rt - 0.5) * 2 + b * (rd - 0.5) * 2) / (a + b)
    #       rt = min(1, cwnd / target_cwnd)
    #       rd = min(1, max(0, min(rd_rtt, rd_queue)))
    #       rd_rtt = 1 - (cwnd - target_cwnd) / (link_min_rtt * link_bandwidth / packet_size)
    #       rd_queue = 1 - (cwnd - target_cwnd - queue_size * f) / (queue_size * (1 - f))
    #   - below_target_cwnd
    #       cwnd <= r' * target_cwnd ? (cwnd / (target_cwnd * r'))^a
    #                                : 1 - min(1, (cwnd - r' * target_cwnd) / (target_cwnd * (1 - r'))) * (1 + b)
    #
    # Note that the delay is computed as max(0, delay - o).
    cc_env_reward_formula: StrEnum(
        "CCEnvRewardFormula",
        "linear, log_ratio, min_throughput, target_cwnd, target_cwnd_shaped, "
        "higher_is_better, above_cwnd, cwnd_range, cwnd_range_soft, cwnd_tradeoff, "
        "below_target_cwnd",
    ) = "log_ratio"
    # Offset to remove from the delay when computing the reward (o).
    cc_env_reward_delay_offset: float = 0.0
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
    # Min ratio of the maximum achievable throughput / target cwnd that we want to reach (r).
    cc_env_reward_min_throughput_ratio: float = 0.9
    # Max ratio of the maximum achievable throughput / target cwnd that we want to reach (r').
    cc_env_reward_max_throughput_ratio: float = 1.0
    # Offset to add to the estimated number of packets in the queue (k).
    cc_env_reward_n_packets_offset: float = 1.0
    # We allow the uplink queue to be filled up to this ratio without penalty (f).
    cc_env_reward_uplink_queue_max_fill_ratio: float = 0.5
    # Whether to take max delay over observations in reward (avg otherwise).
    cc_env_reward_max_delay: bool = True
    # Target fixed cwnd value (only used in 'fixed' env mode).
    cc_env_fixed_cwnd: int = 10
    # Window length (in us) of min RTT filter used to estimate delay.
    # Default is to set it to a very large value (here, 10K seconds) to be sure
    # that we do not under-estimate the delay within an episode.
    cc_env_min_rtt_window_length_us: int = 10_000_000_000
    # Moving average coefficient used to compute the average ACK delay, as
    # well as the average total RTT (including ACK delay).
    # This is the weight of new observations: higher values update the average faster.
    cc_env_ack_delay_avg_coeff: float = 0.1
    # Minimum duration over which the bandwidth is computed (in ms).
    cc_env_bandwidth_min_window_duration_ms: int = 100
    # Factor by which the observation vector should be scaled.
    cc_env_obs_scaling: float = 1.0


def get_actor_data(flags, actor_id):
    """Return the actor's list of jobs and Pantheon environment"""
    # Fetch list of jobs for the current thread.
    jobs = utils.get_jobs(flags, actor_id=actor_id)

    # Each actor (= thread) gets its own seed to sample jobs' parameters.
    resolvers.seed_thread_rng(flags.jobs_seed + actor_id)

    # Environment variables for Pantheon.
    pantheon_env = get_pantheon_env(flags, actor_id)

    return jobs, pantheon_env


def get_empirical_base_rtt(flags, pantheon_env, data_dir, job):
    """
    Empirically obtain the base RTT for the given job.

    This is achieved by running a "dummy" episode with a fixed CWND equal to half
    the theoretical target CWND (so as to ensure there is no added delay caused by
    congestion). We recover the average RTT and use it as base RTT.
    """
    job = copy.deepcopy(job)
    # We need to set the RTT noise to zero in order to obtain a reliable estimate.
    job.rtt_noise_std = 0
    target_cwnd = get_target_cwnd(job)
    flags = copy.deepcopy(flags)
    # Switch to the "fixed CWND" mode with a low CWND (defined as half the target CWND).
    flags.cc_env_mode = "fixed"
    flags.cc_env_fixed_cwnd = int(target_cwnd / 2 + 0.5)
    # Use a small moving average coefficient to have a stable RTT estimate.
    flags.cc_env_ack_delay_avg_coeff = 0.002

    # Obtain command line.
    cmd_tmpl = get_env_cmd(flags, job)
    cmd = [utils.safe_format(c, {"data_dir": data_dir}) for c in cmd_tmpl]
    # `stats_file` is the file where the RTT estimate will become available.
    stats_file = data_dir / "empirical_base_rtt"
    cmd = update_cmd(
        cmd=cmd, flags=flags, job=job, stats_file=stats_file, actor_id=-1, job_count=-1
    )

    # Launch "dummy" episode.
    subprocess.run(cmd, env=pantheon_env)

    if not stats_file.is_file():
        # This can happen if the run failed (e.g., due to connection error).
        logging.warning("No stats file found at %s", stats_file)
        return None

    # Read RTT estimate from stats file.
    with stats_file.open() as f:
        stats_lines = f.readlines()
    logging.info("Read job stats:\n%s", "".join(stats_lines))
    # We obtain this estimate from the last line in the file.
    last_line = None
    for i, line in enumerate(stats_lines):
        if i == 0:  # header
            header = line.strip().split()
            assert len(header) >= 2 and header[1] == "avg_noisy_rtt_no_delay"
        else:
            last_line = line
    if last_line is None:
        logging.warning("No stats found in %s", stats_file)
        return None
    steps, base_rtt = last_line.strip().split()[0:2]
    steps = int(steps)
    if steps < 100:
        # If we do not have enough steps, something wrong happened and we may not have a reliable
        # estimate of the RTT.
        logging.warning("Only reached %s < 100 steps in %s", steps, stats_file)
        return None

    return float(base_rtt)


def get_job_and_cmd(
    flags,
    jobs,
    job_id,
    actor_id,
    data_dir,
    job_count=-1,
    job_info_queue=None,
    learner_has_job_info=None,
):
    """
    Obtain the command line associate to a given job in the list of all jobs.

    The return value is a pair `(job, command_line)`.
    """
    # Select the desired job and resolve its parameters (=> sampling random numbers).

    for attempt in itertools.count():
        job = copy.deepcopy(jobs[job_id])
        OmegaConf.resolve(job)
        logging.debug(f"Sampled job with target cwnd: {get_target_cwnd(job)}")
        if (
            flags.allow_jobs_with_target_cwnd_above_max_cwnd
            or get_target_cwnd(job) <= MAX_CWND
        ):
            break
        if attempt >= 1000:
            raise RuntimeError(f"Unable to sample a valid job after {attempt} attempts")

    # Generate the uplink & donwlink scaled traces (if needed).
    set_scaled_traces(job, data_dir)

    if job_info_queue is not None:
        # Obtain and store the job info in the queue: it will be retrieved by the learner.
        assert job_count >= 0
        task_obs = get_task_obs(flags, job)
        assert learner_has_job_info is not None
        learner_has_job_info.clear()
        job_info_queue.put((actor_id, job_id, job_count, task_obs))
        # Wait for the learner to acknowledge reception.
        while not learner_has_job_info.is_set():
            time.sleep(0.1)

    # Generate the command line (expanding the data directory in the template).
    cmd_tmpl = get_env_cmd(flags, job)
    cmd = [utils.safe_format(c, {"data_dir": data_dir}) for c in cmd_tmpl]
    cmd = update_cmd(
        cmd=cmd, flags=flags, actor_id=actor_id, job=job, job_count=job_count
    )

    return job, cmd


def get_target_cwnd(job):
    """Return the "optimal" CWND value given the job settings"""
    delay = job.delay * 2 / 1000
    bandwidth = TRACE_TO_MAX_THROUGHPUT[job.uplink.trace] * MBYTES_TO_BYTES
    return bandwidth * delay / UDP_SEND_PACKET_LEN


def get_task_obs(flags, job):
    """Return a job's task state (i.e., its parameters)"""
    # Currently we assume the downlink is the same as the uplink.
    assert job.downlink.scaled_trace == job.uplink.scaled_trace

    # Include a one-hot of the job ID (if needed).
    n_train_jobs = get_n_jobs(flags, mode="train")
    job_id_state = make_one_hot(job.job_id, n_train_jobs) if n_train_jobs > 1 else []

    trace_id = TRACE_TO_ID[job.uplink.trace]
    task_obs = torch.tensor(
        job_id_state
        + make_one_hot(trace_id, len(TRACE_TO_ID))
        + [
            # This gives an estimate of the max throughput possible.
            TRACE_TO_MAX_THROUGHPUT[job.uplink.trace]
            * job.uplink.trace_scaling_factor
            / flags.cc_env_norm_bytes,
            job.delay / flags.cc_env_norm_ms,
            np.log(job.uplink.trace_scaling_factor),
            float(job.uplink.loss > 0),
            np.log(job.uplink.loss * 100) if job.uplink.loss > 0 else 0,
            np.log(job.uplink.queue_size_packets / 100),
            np.log(1e-3 + job.rtt_noise_std) - np.log(1e-3),
        ],
        dtype=torch.float32,
    )
    return task_obs * flags.cc_env_obs_scaling


def override_cmd(cmd, overrides):
    """Update `cmd` by overriding some flags"""
    for override_key, override_value in overrides.items():
        pattern = f"{override_key}="
        for i, item in enumerate(cmd):
            # Does this command line item contain the desired key assignment?
            pos = item.find(pattern)
            if pos >= 0:
                start = pos + len(pattern)
                # Find where the key assignment ends.
                end = item.find(" ", start)
                if end < 0:
                    end = len(item)
                cmd[i] = item[0:start] + str(override_value) + item[end:]


def train_run(flags, thread_id, job_info_queue=None, learner_has_job_info=None):
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
    for job_count in itertools.count():
        # Pick a random job to run.
        job_id = next(job_ids)

        # Try to use the RAM disk as data dir, if possible.
        base_folder = utils.get_ramdisk(flags)
        if base_folder is None:
            base_folder = Path(flags.logdir)
        data_dir = base_folder / f"train_tid{thread_id}_run{episode}_expt{job_id}"

        job, cmd = get_job_and_cmd(
            flags,
            jobs=jobs,
            job_id=job_id,
            actor_id=thread_id,
            data_dir=data_dir,
            job_count=job_count,
            job_info_queue=job_info_queue,
            learner_has_job_info=learner_has_job_info,
        )

        can_launch_job = True
        if flags.empirical_base_rtt:
            # Replace the theoretical base RTT with the empirically observed RTT.
            base_rtt = get_empirical_base_rtt(
                flags=flags, pantheon_env=pantheon_env, data_dir=data_dir, job=job
            )
            if base_rtt is None:
                can_launch_job = False
                logging.warning(
                    "Thread: %s, empirical RTT could not be obtained => skipping episode %s",
                    thread_id,
                    episode,
                )
            else:
                logging.info("Obtained empirical base RTT: %f", base_rtt)
                override_cmd(cmd, {"cc_env_base_rtt": str(base_rtt)})

        # Launch the job.
        if can_launch_job:
            p = subprocess.Popen(cmd, env=pantheon_env)
            logging.info(
                "Thread: {}, episode: {}, experiment: {}, process: {}, subprocess: {}, cmd: {}".format(
                    thread_id, episode, job_id, os.getpid(), p.pid, " ".join(cmd)
                )
            )
            p.wait()

            log_func = logging.debug if p.returncode == 0 else logging.warning
            log_func(
                "Thread: %s => process %s exited with return code %s",
                thread_id,
                p.pid,
                p.returncode,
            )

        episode += 1

        # Remove pantheon logs to free up space (may need to retry in case
        # process has not yet fully exited).
        utils.delete_dir(data_dir, max_tries=3, sleep_time=2)

        base_tmp_dir = utils.get_ramdisk(flags)
        if base_tmp_dir is None:
            base_tmp_dir = get_slurm_temporary_dir()
        if base_tmp_dir is not None:
            pantheon_tmp_dir = base_tmp_dir / "pantheon_tmp" / f"actor_{thread_id}"
            if pantheon_tmp_dir.is_dir():
                utils.delete_dir(pantheon_tmp_dir, max_tries=3, sleep_time=2)


def test_run(flags, thread_id, job_info_queue=None, learner_has_job_info=None):
    """
    Thread i runs jobs[i % len(jobs)] flags.test_runs_per_job times.

    NB: currently `test_run` does not fill `job_info_queue` (it assumes that it is
    not needed at test time).
    """
    jobs, pantheon_env = get_actor_data(flags, actor_id=thread_id)

    job_id = thread_id % len(jobs)
    data_dir = Path(flags.logdir) / f"test_expt{job_id}"

    _, cmd = get_job_and_cmd(
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
    analysis_cmd = [flags.analyze_path, f"--data-dir={data_dir}"]
    logging.info(
        "Thread {}, job {}: Running analysis on {}, cmd: {}".format(
            thread_id, job_id, data_dir, " ".join(analysis_cmd)
        )
    )
    p = subprocess.Popen(analysis_cmd, env=pantheon_env)
    p.wait()

    shutil.copyfile(
        data_dir / "pantheon_summary_mean.pdf",
        Path(flags.logdir) / f"test_expt{job_id}.pdf",
    )
    logging.info(
        "Test run finished for thread {}, job {}. Results in {}.".format(
            thread_id, job_id, data_dir
        )
    )


def run_pantheon(
    flags, run_fn, num_threads, learner_has_job_info=None, job_info_queue=None
):
    logging.info("Launching {} threads for {}.".format(num_threads, flags.mode))

    threads = []
    for actor_id in range(num_threads):
        # An independent copy of the config is used in each thread. This is a safeguard
        # against potential thread-safety issues in OmegaConf.
        thread = threading.Thread(
            target=run_fn,
            kwargs=dict(
                flags=copy.deepcopy(flags),
                job_info_queue=job_info_queue,
                thread_id=actor_id,
                learner_has_job_info=None
                if learner_has_job_info is None
                else learner_has_job_info[actor_id],
            ),
        )
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


def get_pantheon_env(flags, actor_id):
    # $PATH override to put python2 first for Pantheon
    result = subprocess.run(
        ["dirname $(which python2)"], shell=True, stdout=subprocess.PIPE
    )
    python2_path = result.stdout.decode("utf-8").strip()
    logging.info("Located python2 in {}".format(python2_path))

    pantheon_env = os.environ.copy()
    pantheon_env["PATH"] = ":".join([python2_path, pantheon_env["PATH"]])
    # Store actor info in env variable: this allows child processes to refer to it in logs.
    pantheon_env["MVFSTRL_ACTOR_ID"] = str(actor_id)
    pantheon_env["MVFSTRL_ACTOR_PID"] = str(os.getpid())
    # Also store path to RAM disk if available, to be used for temporary files.
    ramdisk = utils.get_ramdisk(flags)
    if ramdisk is not None:
        pantheon_env["MVFSTRL_RAM_DISK"] = str(ramdisk)

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


def update_cmd(cmd, flags, actor_id, job, job_count, stats_file=""):
    if flags.mode == "train":
        schemes = "mvfst_rl"
        run_times = 1
    else:  # test mode
        schemes = " ".join(get_test_schemes(flags))
        run_times = flags.test_runs_per_job
        # In test mode we ignore the input job counter and force it to -1.
        # This ensures that the model cannot use job-specific information.
        job_count = -1

    uplink_bandwidth = (
        TRACE_TO_MAX_THROUGHPUT[job.uplink.trace] * job.uplink.trace_scaling_factor
    )
    uplink_queue_size_bytes = job.uplink.queue_size_packets * MM_PACKET_SIZE

    extra_sender_args = " ".join(
        [
            "--cc_env_mode={}".format(flags.cc_env_mode),
            "--cc_env_rpc_address={}".format(flags.server_address),
            "--cc_env_actor_id={}".format(actor_id),
            "--cc_env_job_count={}".format(job_count),
            "--cc_env_model_file={}".format(flags.traced_model),
            "--cc_env_agg={}".format(flags.cc_env_agg),
            "--cc_env_time_window_ms={}".format(flags.cc_env_time_window_ms),
            "--cc_env_fixed_window_size={}".format(flags.cc_env_fixed_window_size),
            "--cc_env_use_state_summary={}".format(flags.cc_env_use_state_summary),
            "--cc_env_history_size={}".format(flags.cc_env_history_size),
            "--cc_env_norm_ms={}".format(flags.cc_env_norm_ms),
            "--cc_env_norm_bytes={}".format(flags.cc_env_norm_bytes),
            "--cc_env_actions={}".format(",".join(flags.cc_env_actions)),
            "--cc_env_uplink_bandwidth={}".format(uplink_bandwidth),
            "--cc_env_uplink_queue_size_bytes={}".format(uplink_queue_size_bytes),
            "--cc_env_base_rtt={}".format(job.delay * 2),
            "--cc_env_reward_delay_offset={}".format(flags.cc_env_reward_delay_offset),
            "--cc_env_reward_formula={}".format(flags.cc_env_reward_formula),
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
            "--cc_env_reward_min_throughput_ratio={}".format(
                flags.cc_env_reward_min_throughput_ratio
            ),
            "--cc_env_reward_max_throughput_ratio={}".format(
                flags.cc_env_reward_max_throughput_ratio
            ),
            "--cc_env_reward_n_packets_offset={}".format(
                flags.cc_env_reward_n_packets_offset
            ),
            "--cc_env_reward_uplink_queue_max_fill_ratio={}".format(
                flags.cc_env_reward_uplink_queue_max_fill_ratio
            ),
            "--cc_env_reward_max_delay={}".format(flags.cc_env_reward_max_delay),
            "--cc_env_fixed_cwnd={}".format(flags.cc_env_fixed_cwnd),
            "--cc_env_min_rtt_window_length_us={}".format(
                flags.cc_env_min_rtt_window_length_us
            ),
            "--cc_env_rtt_noise_std={}".format(job.rtt_noise_std),
            "--cc_env_ack_delay_avg_coeff={}".format(flags.cc_env_ack_delay_avg_coeff),
            "--cc_env_bandwidth_min_window_duration_ms={}".format(
                flags.cc_env_bandwidth_min_window_duration_ms
            ),
            "--cc_env_obs_scaling={}".format(flags.cc_env_obs_scaling),
            "--cc_env_stats_file={}".format(stats_file),
            "-v={}".format(flags.loglevel),
        ]
    )
    return cmd + [
        "--schemes={}".format(schemes),
        "--run-times={}".format(run_times),
        '--extra-sender-args="{}"'.format(extra_sender_args),
    ]


def monitor_from_learner_queue(
    from_learner_queue, ready_event, stop_event, learner_has_job_info
):
    """Monitor messages coming from the learner"""
    while not stop_event.is_set():
        while True:
            try:
                msg = from_learner_queue.get_nowait()
            except queue.Empty:
                break
            if msg.type == "ready":
                ready_event.set()
            elif msg.type == "stop":
                logging.info("Received stop signal from learner")
                stop_event.set()
                break
            elif msg.type == "got_job_info":
                actor_id = msg.data
                assert learner_has_job_info is not None
                learner_has_job_info[actor_id].set()
            else:
                raise TypeError(f"Invalid message type: {msg.type}")

        time.sleep(0.5)


def main(flags, job_info_queue, from_learner_queue=None):

    ready_event = stop_event = msg_monitor = None

    if flags.mode == "train":
        # One thread / pantheon env per actor while training.
        num_threads = flags.num_actors
        run_fn = train_run
    else:
        # One thread per job to test.
        num_threads = utils.get_n_jobs(flags)
        run_fn = test_run

    # Events used to know when the learner has received the information on each thread's job.
    learner_has_job_info = (
        None
        if from_learner_queue is None
        else [threading.Event() for _ in range(num_threads)]
    )

    if from_learner_queue is not None:
        # Start thread monitoring this message queue.
        ready_event = threading.Event()
        stop_event = threading.Event()
        msg_monitor = threading.Thread(
            target=monitor_from_learner_queue,
            kwargs=dict(
                from_learner_queue=from_learner_queue,
                ready_event=ready_event,
                stop_event=stop_event,
                learner_has_job_info=learner_has_job_info,
            ),
        )
        msg_monitor.start()
        while not ready_event.is_set():
            time.sleep(0.1)

    run_pantheon(
        flags=flags,
        run_fn=run_fn,
        num_threads=num_threads,
        job_info_queue=job_info_queue,
        learner_has_job_info=learner_has_job_info,
    )

    if msg_monitor is not None:
        stop_event.set()
        msg_monitor.join()
