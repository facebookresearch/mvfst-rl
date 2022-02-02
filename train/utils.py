# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import hashlib
import inspect
import logging
import os
import pickle
import shutil
import signal
import string
import sys
import time

from collections import namedtuple
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import psutil
from omegaconf import OmegaConf

from train import state


# Used to store information about jobs.
JobInfo = namedtuple("JobInfo", ["job_id", "task_obs"])

# Used to communicate between processes.
Message = namedtuple("Message", ["type", "data"], defaults=(None,))


class _StrEnum(str, Enum):
    """
    "String" enum, such that one can use `if my_enum == "some_string"`.

    Do not instantiate this class directly: instead use the `StrEnum()` function.
    """

    def _generate_next_value_(name, start, count, last_values):
        # This makes it so that the enum's *value* is equal to its *name*.
        # As a result, string comparisons work as expected.
        return name


def StrEnum(name, values):
    """
    Helper function to create a "String" Enum.

    :param name: Name of the Enum being created. The module calling this function
        should not have any other member with the same name (because this function
        will add the created Enum as a member under that name).
    :param values: Values the Enum can take, either as a list of strings, or a
        single string with values separated by whitespaces or commas.
    :return: The corresponding Enum class.

    The motivation for using this helper function vs instantiating `_StrEnum`
    directly is that this function also registers the created Enum within the
    calling module. This makes it possible to pickle / unpickle this Enum as
    long as it is created at import time. The typical use case is within
    structured configs, as follows:

        @dataclass
        class MyConfig:
            color: StrEnum("Color", "red, green, blue") = "blue"

    Such a config (after being processed through Hydra / OmegaConf) may be used
    with string comparisons:

        if cfg.color == "red":
            ...
        elif cfg.color == "green":
            ...
        ...

    NB: a function is used instead of adding this logic to `_StrEnum.__init__()`
    because it turns out to be tricky to override an Enum constructor.
    """
    enum = _StrEnum(name, values)
    # Obtain the module this function was called from.
    stack = inspect.stack()[1]
    src_module = inspect.getmodule(stack[0])
    # Add the generated Enum as member of this module.
    assert not hasattr(
        src_module, name
    ), f"module {src_module} already has a member '{name}'"
    setattr(src_module, name, enum)
    # We also need to set the Enum's module accordingly. As a result the unpickling
    # sequence of an instance of this Enum will be as follows:
    #   1. Identify this Enum's module as `src_module`
    #   2. Import `src_module`
    #   3. During import, this function (= `StrEnum()`) is called
    #   4. During this call, the Enum class is created and registered into `src_module`
    #   5. Once `src_module` is imported, the Enum class is obtained from it and the
    #      Enum instance can be created
    enum.__module__ = src_module.__name__
    return enum


def add_to_path(path):
    """
    Add a path to `sys.path` / PYTHONPATH env variable.

    We also add it to PYTHONPATH so that when unpickling an object (e.g. with
    submitit) Python can find the required packages even if the module
    modifying `sys.path` has not been imported yet.
    """
    if path not in sys.path:
        sys.path.append(path)
    try:
        python_path = os.environ["PYTHONPATH"]
    except KeyError:
        os.environ["PYTHONPATH"] = path
    else:
        tokens = python_path.split(os.pathsep)
        if path not in tokens:
            os.environ["PYTHONPATH"] += os.pathsep + path


def balanced_randints(rng, n: int) -> Generator[int, None, None]:
    """
    Generate random numbers from 0 to n-1 in a balanced way.

    More precisely, this consists in the following steps:
        1. Shuffle a sequence of numbers from 0 to n-1
        2. Yield elements in this shuffled sequence
        3. Repeat from step 1
    """
    indices = list(range(n))
    while True:
        rng.shuffle(indices)
        for i in indices:
            yield i


def ceil_int_div(n: int, d: int) -> int:
    """Return ceil(n / d)"""
    # See https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(n // -d)


def clear_ramdisk(flags) -> None:
    """
    Clear data from ramdisk.

    This clears both:
        * Data from the current job
        * Data from other jobs whose owner's process has died
    """
    base_dir, job_dir = get_ramdisk_paths(flags)
    if job_dir is None:
        return

    delete_dir(job_dir)

    if not base_dir.is_dir():
        return

    try:
        for other in base_dir.iterdir():
            if not other.is_dir():
                logging.warning("Found a non-directory item in RAM disk folder")
                continue
            pid_path = other / "owner_pid"
            if not pid_path.is_file():
                # Note: this might happen in case of concurrent processes.
                logging.warning("Found a directory in RAM disk with no PID file")
                continue
            try:
                with pid_path.open() as f:
                    pid = int(f.read())
            except Exception:
                logging.warning(f"Failed to read PID from: {pid_path}")
                continue
            try:
                psutil.Process(pid)
            except psutil.NoSuchProcess:
                logging.info(f"Deleting old RAM disk folder from PID {pid}: {other}")
                delete_dir(other)

    except Exception:
        logging.exception(f"Failed to clean up base RAM disk folder: {base_dir}")


def default_empty_dict():
    """
    Helper function to declare a dataclass field whose default value is an empty dict.
    """
    return field(default_factory=dict)


def default_empty_list():
    """
    Helper function to declare a dataclass field whose default value is an empty list.
    """
    return field(default_factory=list)


def default_list(lst):
    """
    Helper function to declare a dataclass field whose default value is the list `lst`.
    """
    # We make a copy of `lst` to be sure it is not accidentally shared.
    return field(default_factory=lambda: list(lst))


def generate_scaled_trace(
    base_path: Path,
    scaled_path: Path,
    scaling_factor: float,
    min_scaled_trace_size: int,
) -> None:
    """
    Generate a scaled trace from a base trace, given a scaling factor.

    :param base_path: Path to the input base trace.
    :param scaled_path: Path where the output scaled trace should be saved.
    :param scaling_factor: Scaling factor for trace timestamps. A value > 1 means that
        the trace is sped-up by this factor (a value <1 means slowing down the trace).
    :param min_scaled_trace_size: Minimum size (in number of rows) of the output scaled
        trace file. If the input trace has fewer rows, we duplicate it until this minimum
        number of rows is reached. This makes the scaled trace more accurate, as
        otherwise the rounding to the nearest millisecond may cause issues. For instance
        if the input trace only contains "1" and we want to speed it up by 30% (i.e.,
        the scaling factor is 1.3), we cannot just have a single integer number anymore.
    """
    start_time = time.perf_counter()

    # Read the base trace.
    with base_path.open() as f:
        base_trace = f.readlines()
    data = np.array([int(x) for x in base_trace], dtype=np.float64)

    # Duplicate the data (if needed) until we reach the desired number of rows.
    n_dup = ceil_int_div(min_scaled_trace_size, len(data))
    if n_dup > 1:
        # `offsets` is used to shift the timestamps of the duplicated copies.
        # It looks like this for an input trace with three rows ending at 100ms:
        #   [0, 0, 0, 100, 100, 100, 200, 200, 200, 300, 300, 300, ...]
        offsets = np.repeat(np.arange(n_dup) * data[-1], len(data))
        # Copy the data multiple times, shifting the copies by `offsets`.
        data = np.tile(data, n_dup) + offsets

    # Scale timestamps based on the scaling factor.
    assert scaling_factor > 0
    data /= scaling_factor

    # Save the resulting trace.
    scaled_path.parent.mkdir(parents=True, exist_ok=True)
    with scaled_path.open("w") as f:
        # We round floats to nearest integer.
        f.write("\n".join([f"{t:.0f}" for t in data]))

    duration = time.perf_counter() - start_time
    logging.info(f"Generated scaled trace in {duration:.2f}s: {scaled_path}")


def get_actions(num_actions):
    ACTIONS = {
        5: ["0", "/2", "-10", "+10", "*2"],
        7: ["0", "/2", "/1.5", "-10", "+10", "*1.5", "*2"],
        9: ["0", "/2", "/1.5", "/1.25", "-10", "+10", "*1.25", "*1.5", "*2"],
        11: [
            "0",
            "/5",
            "/2",
            "/1.5",
            "/1.25",
            "-10",
            "+10",
            "*1.25",
            "*1.5",
            "*2",
            "*5",
        ],
    }
    assert num_actions in ACTIONS, "Unsupported num_actions"
    return ACTIONS[num_actions]


def get_jobs(flags, mode=None, actor_id=None):
    from train import pantheon_env  # lazy import to avoid circular dependencies

    mode = flags.mode if mode is None else mode
    if mode == "train":
        if actor_id is None or actor_id < flags.num_actors_train:
            job_ids = flags.train_job_ids
        else:
            job_ids = flags.eval_job_ids
    elif mode == "test":
        job_ids = flags.test_job_ids
    else:
        raise ValueError(mode)

    return pantheon_env.get_jobs_to_perform(
        flags=flags, job_ids=job_ids, actor_id=actor_id
    )


def get_n_jobs(flags, mode=None, actor_id=None):
    return len(get_jobs(flags, mode=mode, actor_id=actor_id))


def get_observation_length(history_size, num_actions):
    # The observation contains:
    # - state summary stats (5 * num_fields) (5 because sum / mean / std / min /max)
    # - history_size * (one-hot actions + cwnd)
    # This formula must be kept in synch with the one used in
    #   CongestionControlEnv::Observation::toTensor()
    return 5 * state.Field.NUM_FIELDS.value + history_size * (num_actions + 1)


def get_ramdisk(flags, create: bool = False) -> Optional[Path]:
    """Return path to RAM disk for this job, if any"""
    base_dir, job_dir = get_ramdisk_paths(flags)

    if job_dir is None:
        return None

    if job_dir.is_dir():
        return job_dir

    if not create:
        return None

    # Try to create the RAM disk folder.
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.warning(f"Unable to created RAM disk folder: {job_dir}")
        return None

    # Store the current PID: this way, another process may know that it is
    # ok to delete this RAM disk if this process has died.
    with (job_dir / "owner_pid").open("w") as f:
        f.write(f"{os.getpid()}")


def get_ramdisk_paths(flags) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return paths for RAM disk folders.

    The return value is a pair containing two paths:
        - The base path for all mvfst-rl RAM disk folders
        - The RAM disk folder associated to the current job

    If there is no RAM disk available then both paths are `None`.
    """
    shm = Path("/dev") / "shm"
    if not shm.is_dir():
        return None, None

    base_dir = shm / "mvfst-rl.tmp"
    # We use the hash of the logdir as folder name.
    logdir_hash = hashlib.sha256(str(flags.base_logdir).encode("ascii")).hexdigest()

    return base_dir, base_dir / logdir_hash


def get_slurm_temporary_dir() -> Optional[Path]:
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id is None:
        return None
    job_dir = Path("/scratch") / "slurm_tmpdir" / slurm_job_id
    return job_dir if job_dir.is_dir() else None


def load_traced_model_flags(model: Path) -> Dict[str, any]:
    """Load flags associated to a given traced model"""
    flags_pkl = model.with_suffix(".flags.pkl")
    assert flags_pkl.is_file(), f"missing flags for model {model}"
    with flags_pkl.open("rb") as f:
        flags_dict = pickle.load(f)
    return OmegaConf.create(flags_dict)


def make_one_hot(i: int, n: int) -> List[int]:
    """Return a one-hot list of size `n` with a one at position `i`"""
    assert n > 1, "you probably do not want a one-hot of size <= 1"
    one_hot = [0] * n
    one_hot[i] = 1
    return one_hot


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# format 'format_string' but ignore keys that do not exist in 'key_dict'
def safe_format(format_string, key_dict):
    return string.Formatter().vformat(format_string, (), SafeDict(key_dict))


def delete_dir(dir_path, max_tries=1, sleep_time=1):
    """Delete a directory (with potential retry mechanism)"""
    if not dir_path.exists():
        return

    for i in range(max_tries):
        start_time = time.perf_counter()
        try:
            shutil.rmtree(dir_path)
        except Exception:
            if i == max_tries - 1:
                logging.warning("Failed to delete dir (giving up): %s", dir_path)
                break
            else:
                logging.info("Failed to delete dir (will try again): %s", dir_path)
                time.sleep(sleep_time)
        else:
            duration = time.perf_counter() - start_time
            logging.info("Deleted dir (in %.3fs): %s", duration, dir_path)
            break


# Mostly copied from https://psutil.readthedocs.io/en/latest/#kill-process-tree
def kill_proc_tree(
    pid, sig=signal.SIGKILL, include_parent=True, timeout=None, on_terminate=None
):
    """Kill a process tree (including grandchildren) with signal
    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
    called as soon as a child terminates.
    """
    assert pid != os.getpid(), "won't kill myself"
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    if include_parent:
        children.append(parent)
    for p in children:
        try:
            p.send_signal(sig)
        except psutil.NoSuchProcess:
            # It seems possible (in rare cases) for the process to have
            # terminated already, triggering this exception.
            continue

    gone, alive = psutil.wait_procs(children, timeout=timeout, callback=on_terminate)
    return (gone, alive)


def utc_date():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M")
