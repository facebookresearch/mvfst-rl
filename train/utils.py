# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import inspect
import itertools
import logging
import os
import shutil
import signal
import socket
import string
import sys
import time
import yaml

from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Optional

import psutil

from train.constants import SRC_DIR, PANTHEON_ROOT, EXPERIMENTS_CFG


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


def get_cpus_per_task(mode, num_actors, test_job_ids, test_after_train, max_jobs):
    """Return number of CPUs to reserve given the current settings"""
    from train import pantheon_env  # lazy import to avoid circular dependencies

    # Reserve 2 CPUs per Pantheon "thread" (at least 4 total). During training,
    # the number of threads is equal to the number of actors, while during
    # testing it is equal to the number of jobs.
    n_cpus_min = 4
    n_threads_train = n_threads_test = 0

    assert mode in ["train", "test"]
    if mode == "train":
        n_threads_train = num_actors
    if mode == "test" or test_after_train:
        jobs = pantheon_env.get_jobs_to_perform(test_job_ids, max_jobs)
        n_threads_test = len(jobs)

    return max(n_cpus_min, n_threads_train * 2, n_threads_test * 2)


def get_jobs(flags, mode=None):
    from train import pantheon_env  # lazy import to avoid circular dependencies

    mode = flags.mode if mode is None else mode
    if mode == "train":
        job_ids = flags.train_job_ids
    elif mode == "test":
        job_ids = flags.test_job_ids
    else:
        raise ValueError(mode)

    return pantheon_env.get_jobs_to_perform(job_ids, flags.max_jobs)


def get_n_jobs(flags, mode=None):
    return len(get_jobs(flags, mode=mode))


def get_observation_length(history_size, num_actions):
    # The observation contains:
    # - state summary stats (5 * 20) (5 because sum / mean / std / min /max)
    # - history_size * (one-hot actions + cwnd)
    # - job ID
    return 100 + history_size * (num_actions + 1) + 1


def get_slurm_constraint(partition: str, gpus_per_node: int) -> Optional[str]:
    """Return the constraint to be used by the `submitit_slurm` launcher"""
    if partition in ["priority", "learnfair"] and gpus_per_node <= 2:
        # If we are on the right environment, use constraint "gpu2".
        host = socket.gethostname()
        if host.startswith("devfair") and len(host) == 11:  # H2?
            return "gpu2"
    return None


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


def parse_experiments():
    with open(EXPERIMENTS_CFG) as cfg:
        return yaml.full_load(
            safe_format(
                cfg.read(), {"src_dir": SRC_DIR, "pantheon_root": PANTHEON_ROOT}
            )
        )


def delete_dir(dir_path, max_tries=1, sleep_time=1):
    """Delete a directory (with potential retry mechanism)"""
    if not os.path.exists(dir_path):
        return

    for i in range(max_tries):
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
            logging.info("Deleted dir: %s", dir_path)
            break


expt_cfg = parse_experiments()
meta = expt_cfg["meta"]


def expand_matrix(matrix_cfg):
    input_list = []
    for variable, value_list in matrix_cfg.items():
        input_list.append([{variable: value} for value in value_list])

    ret = []
    for element in itertools.product(*input_list):
        tmp = {}
        for kv in element:
            tmp.update(kv)
        ret.append(tmp)

    return ret


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
