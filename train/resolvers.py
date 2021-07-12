# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import socket
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf

from train.utils import get_n_jobs


# Thread-local storage, used to store a thread-specific RNG to ensure that parallel
# threads do not impact each other when sampling random numbers.
thread_data = threading.local()


def choice(*args: Any) -> Any:
    """
    Return a random item from the arguments list.

    If the last argument is a list, it is interpreted as weights associated to the
    previous arguments. For instance:
        choice(a, b, c, [0.1, 0.3, 0.6])
    will sample either a, b or c with respective probabilities 0.1, 0.3 and 0.6.
    """
    if len(args) > 1 and isinstance(args[-1], list):
        weights = args[-1]
        n_options = len(args) - 1
        assert all(w >= 0 for w in weights), "weights must be non-negative"
        assert abs(sum(weights) - 1) < 1e-6, "weights should sum to one"
        assert len(weights) == len(args) - 1, "incorrect number of weights"
    else:
        weights = None
        n_options = len(args)

    idx = thread_data.rng.choice(n_options, p=weights)
    return args[idx]


def get_cpus_per_task(mode, num_actors, test_after_train, *, _root_):
    """Return number of CPUs to reserve given the current settings"""
    # Reserve 2 CPUs per Pantheon "thread" (at least 4 total). During training,
    # the number of threads is equal to the number of actors, while during
    # testing it is equal to the number of jobs.
    n_cpus_min = 4
    n_threads_train = n_threads_test = 0

    assert mode in ["train", "test"]
    if mode == "train":
        n_threads_train = num_actors
    if mode == "test" or test_after_train:
        n_threads_test = get_n_jobs(flags=_root_, mode="test")

    return max(n_cpus_min, n_threads_train * 2, n_threads_test * 2)


def get_slurm_constraint(partition: str, gpus_per_node: int) -> Optional[str]:
    """Return the constraint to be used by the `submitit_slurm` launcher"""
    if partition in ["priority", "learnfair"] and gpus_per_node <= 2:
        # If we are on the right environment, use constraint "gpu2".
        host = socket.gethostname()
        if host.startswith("devfair") and len(host) == 11:  # H2?
            return "gpu2"
    return None


def minus(a, b):
    return a - b


def resolve_path(path: str, relative_to: Optional[str] = None) -> str:
    """
    Obtain an absolute path, resolving any symlinks.

    This is similar to `Path(path).resolve()` except that if `relative_to` is
    provided and `path` is a relative path, then the returned path is obtained
    from the concatenation of `realative_to` with `path`.
    """

    p = Path(path)
    if relative_to is not None and not p.is_absolute():
        p = Path(relative_to) / p
    return str(p.resolve())


def uniform_log(low: float, high: float, median: Optional[float] = None) -> float:
    """
    Sample a number between `low` and `high` uniformly in log-space.

    The optional `median` parameter indicates the target value for the median of the
    resulting distribution. It must be above `low` and below the average of `low` and
    `high`. If not provided, the median is such that, in log-space, `low` is mapped
    to 0, `high` is mapped to 1 and the median is mapped to 0.5.
    """
    if high == low:
        return low
    assert high > low

    # Compute the offset to apply to `low` and `high` before taking the log.
    if median is None:
        # This ensures that log(low + offset) == 0.
        offset = 1.0 - low
    else:
        assert median > low, "median must be higher than the lower value"
        denominator = low + high - 2 * median
        assert denominator > 0, "median must be lower than the mean of low and high"
        # This "magic" formula ensures that when sampling uniformly in the interval
        # [log(low + offset), log(high + offset)], the middle point yields the desired
        # median value.
        offset = (median ** 2 - low * high) / denominator

    # Sample uniformly in log space.
    u = thread_data.rng.uniform(np.log(low + offset), np.log(high + offset))

    # Take exponential and cancel out the offset to yield the sampled value.
    val = np.exp(u) - offset

    assert low <= val <= high, (low, val, high)
    return float(val)


def uniform_log_int(low: int, high: int, median: Optional[int] = None) -> int:
    """
    Similar to `uniform_log()` but for integer values.
    """
    # We sample a float then cast to nearest integer value.
    float_val = uniform_log(
        low=float(low),
        high=float(high),
        median=None if median is None else float(median),
    )
    return int(float_val + 0.5)


# The functions below are related to resolvers but are not resolvers themselves.


def seed_thread_rng(seed: int) -> None:
    """Seed the RNG used by resolvers sampling random numbers"""
    thread_data.rng = np.random.default_rng(seed)


# Register resolvers.
OmegaConf.register_new_resolver("choice", choice)
OmegaConf.register_new_resolver("get_cpus_per_task", get_cpus_per_task)
OmegaConf.register_new_resolver("get_slurm_constraint", get_slurm_constraint)
OmegaConf.register_new_resolver("minus", minus)
OmegaConf.register_new_resolver("resolve_path", resolve_path)
OmegaConf.register_new_resolver("uniform_log", uniform_log)
OmegaConf.register_new_resolver("uniform_log_int", uniform_log_int)
