# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import concurrent.futures
import copy
import csv
import json
import logging
import math
import os
import pickle
import sys
import time
import timeit

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402

import numpy as np
import torch

from omegaconf import OmegaConf

from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from train import state, utils
from train.constants import TORCHBEAST_ROOT, UDP_SEND_PACKET_LEN
from train.models import SimpleNet
from train.utils import StrEnum, get_jobs, get_n_jobs

sys.path.append(TORCHBEAST_ROOT)

from atari import vtrace
from torchbeast import queue
import nest
import torchbeast


@dataclass
class ConfigLearner(Dict[str, Any]):
    # Model output settings.

    # Experiment id (default: None).
    xpid: Optional[str] = None
    # File to write checkpoints to.
    checkpoint: str = "checkpoint.tar"

    # Model settings.

    # Length of the observation vector to be fed into the model.
    observation_length: int = -1  # will be set at init time
    # Hidden size in FC model.
    hidden_size: int = 1024
    # Number of actions output by the policy.
    num_actions: int = 5
    # Total environment steps to train for".
    total_steps: int = 1_000_000
    # Learner batch size.
    batch_size: int = 8
    # Inference batch size.
    inference_batch_size: int = 2
    # The unroll length (time dimension).
    unroll_length: int = 80
    # Whether to use LSTM in agent model.
    use_lstm: bool = True
    # Initial random seed for torch.random.
    seed: int = 1234
    # If True, then we pretend the episode would continue after it ended,
    # by bootstrapping with the last state's value..
    end_of_episode_bootstrap: bool = False
    # When training on more than one job (from `experiments.yml`), these flags
    # indicate whether the actor / critic should have access to the job ID.
    use_job_id_in_actor: bool = False
    use_job_id_in_critic: bool = False

    # GALA settings.

    # If > 1, GALA mode is enabled.
    num_gala_agents: int = 1
    # Number of peers to communicate with in each iteration.
    num_gala_peers: int = 1
    # Max amount of message staleness for local gossip.
    sync_freq: int = 0

    # Loss settings.

    # Entropy cost/multiplier.
    entropy_cost: float = 0.01
    # Baseline cost/multiplier.
    baseline_cost: float = 0.5
    # Discounting factor
    discounting: float = 0.99
    # Reward clipping mechanism.
    reward_clipping: StrEnum(
        "RewardClipping", "abs_one, soft_asymmetric, none"
    ) = "none"
    # Whether to enable adaptive reward normalization. If True, then rewards are rescaled as
    #   reward = (reward - mu) / std * sqrt(1 - gamma^2)
    # where `mu` and `std` are respectively the running mean and standard deviation of
    # observed rewards. The motivation for this formula is that, assuming iid rewards, the
    # return over an infinite horizon would have mean 0 and variance 1.
    reward_normalization: bool = True
    # If `True`, then the running mean / std of observed rewards used when
    # `reward_normalization` is enabled are computed independently for each training
    # job (as defined from `ConfigEnv.train_job_ids`). This is particularly useful
    # when wildly different network conditions are seen during training.
    reward_normalization_stats_per_job: bool = True
    # The coefficient used in the moving averages keeping track of the mean and standard
    # deviation of the rewards when `reward_normalization` is True.
    # Denoting this coefficient as `c`, the mean `mu_` is updated on each batch by:
    #   mu_ = (1-c)^n * mu_ + (1 - (1-c)^n) * avg_batch(reward - offset)
    # where `n` is the total number of rewards observed in the batch, and `avg_batch()`
    # denotes the average across the batch. Note that:
    #   - We use `^n` to make the behavior less dependent on the batch size
    #   - `offset` is a fixed value that is subtracted from rewards to stabilize numeric
    #     operations (see below for more details on how it is used and computed). The `mu`
    #     used in the reward normalization formula is obtained by `mu = offset + mu_`.
    # In order to keep track of the standard deviation, we also maintain a moving average
    # of the squared (shifted) rewards, as:
    #   sq = (1-c)^n * sq + (1 - (1-c)^n) * avg_batch((reward - offset) ** 2)
    # The standard deviation `std` can be obtained from the variance of `reward - offset`:
    #   std^2 = V[reward - offset]
    #         = E[(reward - offset)^2] - E[reward - offset]^2
    #         = sq - mu_^2
    # Since we use moving averages, there is a risk that a poor initialization may slow down
    # initial convergence towards the statistics we want to track. As a result, during the
    # first `1 / c` steps, `mu` and `std` are computed simply by keeping track of all rewards
    # observed since the beginning of training, and calculating their mean and standard
    # deviation. The switch to the moving averages is triggered after `1 / c` steps, where
    # we initialize the following variables, based on all rewards observed so far:
    #   * mu_ = 0
    #   * offset = average of all rewards
    #   * sq = (standard deviation of all rewards) ** 2
    reward_normalization_coeff: float = 1e-4
    # Small epsilon used to lower bound the variance, to avoid a variance <= 0.
    reward_normalization_var_eps: float = 1e-8

    # Optimizer settings.

    # Learning rate.
    learning_rate: float = 1e-5
    # "Learning rate decay method.
    learning_rate_decay: StrEnum("LearningRateDecay", "linear, none") = "linear"
    # RMSProp smoothing constant.
    alpha: float = 0.99
    # RMSProp momentum.
    momentum: float = 0
    # RMSProp epsilon.
    epsilon: float = 0.01
    # Global gradient norm clip. Set to 0 to disable.
    grad_norm_clipping: float = 0

    # Hardware settings.

    # Device for learning.
    learner_device: str = "cuda:0"
    # Device for inference.
    inference_device: str = "cuda:1"


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def compute_baseline_loss(advantages):
    # NB: these "advantages" are not the same as the advantages used in
    # `compute_policy_gradient_loss()`, as they are based on `vs` rather
    # than `rs + vs+1`. See top right of p.4 in IMPALA's paper.
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    # In IMPALA's paper this loss yields the gradient described at end of
    # section 4 ("and the policy parameters w in the direction of the policy
    # gradient").
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )  # (T * N)
    cross_entropy = cross_entropy.view_as(advantages)  # (T, N)
    return torch.sum(cross_entropy * advantages.detach())


def compute_metrics(
    flags,
    learner_outputs,
    actor_outputs,
    env_outputs,
    last_actions,
    reward_stats=None,
    end_of_episode_bootstrap=False,
):
    """
    Compute various metrics (including in particular the loss being optimized).

    :param learner_outputs: A dictionary holding the following tensors, where
        `T` is the unroll length and `N` the batch size (number of actors):
        * "action": (T + 1, N)
        * "baseline": (T + 1, N)
        * "policy_logits": (T + 1, N, num_actions)
    :param actor_outputs: Similar to `learner_outputs`.
    :param env_outputs: A triplet of observation (T + 1, N, obs_dim), reward
        (T + 1, N) and done (T + 1, N) tensors.
    :param last_actions: not used
    :param reward_stats: Must be provided when reward normalization is enabled.
        This dictionary holds statistics on the observed rewards.
    """
    del last_actions  # Only used in model.
    # Estimated value of the last state in the rollout (N).
    bootstrap_value = learner_outputs["baseline"][-1]

    # Move from obs[t] -> action[t] to action[t] -> obs[t].
    # After this step all tensors have shape (T, N, ...)
    actor_outputs = nest.map(lambda t: t[:-1], actor_outputs)
    rewards, done = nest.map(lambda t: t[1:], env_outputs[1:])
    learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

    if flags.reward_clipping == "abs_one":
        rewards = torch.clamp(rewards, -1, 1)
    elif flags.reward_clipping == "soft_asymmetric":
        squeezed = torch.tanh(rewards / 5.0)
        # Negative rewards are given less weight than positive rewards.
        rewards = torch.where(rewards < 0, 0.3 * squeezed, squeezed) * 5.0
    elif flags.reward_clipping != "none":
        raise NotImplementedError(flags.reward_clipping)

    if flags.reward_normalization:
        train_job_id = env_outputs[0][1:][:, :, -1].long()
        normalize_rewards(flags, train_job_id, rewards, reward_stats)

    discounts = (~done).float() * flags.discounting

    actor_logits = actor_outputs["policy_logits"]  # (T, N, num_actions)
    learner_logits = learner_outputs["policy_logits"]  # (T, N, num_actions)
    actions = actor_outputs["action"]  # (T, N)
    baseline = learner_outputs["baseline"]  # (T, N)

    vtrace_returns = vtrace.from_logits(
        behavior_policy_logits=actor_logits,
        target_policy_logits=learner_logits,
        actions=actions,
        discounts=discounts,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        end_of_episode_bootstrap=end_of_episode_bootstrap,
        done=done,
    )

    pg_loss = compute_policy_gradient_loss(
        learner_logits, actions, vtrace_returns.pg_advantages
    )
    baseline_loss = compute_baseline_loss(
        # NB: if `end_of_episode_bootstrap` is True, then `vs` at end of episode
        # is equal to the baseline, contributing to a zero cost. This is what we
        # want, as we cannot learn from that step without knowing the next state.
        vtrace_returns.vs - baseline
    )
    entropy_loss = compute_entropy_loss(learner_logits)

    # Total entropy if the learner was outputting a uniform distribution over actions
    # (used for normalization in the reported metrics).
    batch_total_size = learner_logits.shape[0] * learner_logits.shape[1]
    uniform_entropy = batch_total_size * np.log(flags.num_actions)

    return {
        "loss/total": pg_loss
        + flags.baseline_cost * baseline_loss
        + flags.entropy_cost * entropy_loss,
        "loss/pg": pg_loss,
        "loss/baseline": baseline_loss,
        "loss/normalized_neg_entropy": entropy_loss / uniform_entropy,
        "critic/baseline/mean": torch.mean(baseline),
        "critic/baseline/min": torch.min(baseline),
        "critic/baseline/max": torch.max(baseline),
    }


@torch.no_grad()
def normalize_rewards(flags, train_job_id, rewards, reward_stats):
    """
    Compute the normalized rewards, updating `reward_stats` at the same time.

    :param flags: Config flags.
    :param train_job_id: Tensor (T, N) of integers providing the training job ID
        associated to the corresponding reward in `rewards`.
    :param rewards: Tensor (T, N) of observed rewards. This tensor is updated
        in-place.
    :param reward_stats: The dictionay holding reward stats.
        If `reward_normalization_stats_per_job` is False, it has a single key (0),
        and otherwise it has one key per train job ID. Each key is associated to
        a dictionary with the following key / value pairs:
            - Before `1/reward_normalization_coeff` rewards have been seen (for the
              corresponding key):
                * "initial_rewards": a tensor containing all rewards since so far
            - After `1/reward_normalization_coeff` rewards have been seen:
                * "offset": applied to rewards to shift them
                * "mean": running mean of (shifted) rewards
                * "mean_squared": running mean of (shifted) squared rewards
    """
    rewards_flat = rewards.flatten()
    is_job_id = None
    if flags.reward_normalization_stats_per_job:
        # Split rewards according to their train job ID.
        train_job_id_flat = train_job_id.flatten()  # [T * N]
        all_job_ids = torch.unique(train_job_id_flat).view(-1, 1)  # [M, 1]
        if len(all_job_ids) == 1:
            # Optimization if there is a single job ID.
            rewards_per_job = {all_job_ids[0].item(): rewards_flat}
        else:
            is_job_id = train_job_id_flat == all_job_ids  # [M, T * N]
            rewards_per_job = {
                job_id.item(): rewards_flat[match]
                for job_id, match in zip(all_job_ids, is_job_id)
            }
    else:
        rewards_per_job = {0: rewards_flat}

    normalized_rewards = []
    for job_id, job_rewards in rewards_per_job.items():
        job_stats = reward_stats[job_id]
        if "initial_rewards" in job_stats:
            # Use regular mean / std instead of moving averages.
            initial_rewards = job_stats["initial_rewards"] = torch.cat(
                (job_stats["initial_rewards"], job_rewards)
            )
            mean_rewards = initial_rewards.mean()
            std_rewards = max(
                math.sqrt(flags.reward_normalization_var_eps), initial_rewards.std()
            )
            n_init = int(1 / flags.reward_normalization_coeff + 0.5)
            if len(initial_rewards) >= n_init:
                # Switch to moving averages.
                del job_stats["initial_rewards"]
                job_stats["mean"] = 0.0
                job_stats["offset"] = mean_rewards
                job_stats["mean_squared"] = std_rewards ** 2
                logging.info(
                    f"Reward statistics (job_id={job_id}) after observing the first "
                    f"{len(initial_rewards)} rewards: mean = {mean_rewards}, std = {std_rewards}"
                )
        else:
            # Use moving averages.
            mean_rewards = job_stats["mean"]
            mean_squared_rewards = job_stats["mean_squared"]
            offset = job_stats["offset"]

            # Shift rewards by offset.
            rewards_shifted = job_rewards - offset
            shifted_mean = rewards_shifted.mean()

            # Update moving averages.
            n = len(rewards_shifted)
            alpha = (1 - flags.reward_normalization_coeff) ** n
            mean_rewards = job_stats["mean"] = (
                alpha * mean_rewards + (1 - alpha) * shifted_mean
            )
            shifted_squared_mean = (rewards_shifted ** 2).mean()
            mean_squared_rewards = job_stats["mean_squared"] = (
                alpha * mean_squared_rewards + (1 - alpha) * shifted_squared_mean
            )
            var_rewards = max(
                flags.reward_normalization_var_eps,
                mean_squared_rewards - mean_rewards ** 2,
            )

            # Compute the mean / std used for reward normalization.
            mean_rewards = offset + mean_rewards
            std_rewards = math.sqrt(var_rewards)

        # Apply the reward normalization formula.
        new_job_rewards = (job_rewards - mean_rewards) * (
            math.sqrt(1 - flags.discounting ** 2) / std_rewards
        )
        assert new_job_rewards.dtype == torch.float32
        normalized_rewards.append(new_job_rewards)

    # Update the `rewards` input tensor with normalized rewards.
    if len(normalized_rewards) == 1:
        # Single job ID.
        rewards_flat[:] = normalized_rewards[0]
    else:
        assert is_job_id is not None
        for job_normalized_rewards, match in zip(normalized_rewards, is_job_id):
            rewards_flat[match] = job_normalized_rewards


class Rollouts:
    """
    Storage class holding rollout data.

    Each type of data is associated with a tensor of shape N x L x ... where:
        - `N` is the number of actors
        - `L` is the rollout length + 1 (+1 because even for an unroll length
          equal to 1, we need data from two timesteps, so as to know both the
          start and end points)
        - `...` is the shape of this data for a single agent at a single timestep
          (ex: empty for actions, 1D for observations, etc)
    """

    def __init__(self, timestep, unroll_length, num_actors, num_overlapping_steps=0):
        self._full_length = num_overlapping_steps + unroll_length + 1
        self._num_overlapping_steps = num_overlapping_steps

        N = num_actors
        L = self._full_length

        self._state = nest.map(
            lambda t: torch.zeros((N, L) + t.shape, dtype=t.dtype), timestep
        )

        self._index = torch.zeros([N], dtype=torch.int64)

    def append(self, actor_ids, timesteps):
        assert len(actor_ids) == len(
            actor_ids.unique()
        ), f"Duplicate actor ids: {list(sorted(actor_ids))}"
        for s in nest.flatten(timesteps):
            assert s.shape[0] == actor_ids.shape[0], "Batch dimension don't match"

        curr_indices = self._index[actor_ids]

        for s, v in zip(nest.flatten(self._state), nest.flatten(timesteps)):
            s[actor_ids, curr_indices] = v

        self._index[actor_ids] += 1

        return self._complete_unrolls(actor_ids)

    def _complete_unrolls(self, actor_ids):
        """Obtain unrolls that have reached the desired length"""
        actor_indices = self._index[actor_ids]

        actor_ids = actor_ids[actor_indices == self._full_length]
        unrolls = nest.map(lambda s: s[actor_ids], self._state)

        # Reset state of completed actors to start from the end of the previous
        # ones (NB: since `unrolls` is a copy it is ok to do it in place).
        j = self._num_overlapping_steps + 1
        for s in nest.flatten(self._state):
            s[actor_ids, :j] = s[actor_ids, -j:]

        self._index.scatter_(0, actor_ids, 1 + self._num_overlapping_steps)

        return actor_ids, unrolls

    def reset(self, actor_ids):
        j = self._num_overlapping_steps
        self._index.scatter_(0, actor_ids, j)

        for s in nest.flatten(self._state):
            # .zero_() doesn't work with tensor indexing?
            s[actor_ids, :j] = 0


class StructuredBuffer:
    def __init__(self, state):
        self._state = state

    def get(self, ids):
        return nest.map(lambda s: s[ids], self._state)

    def set(self, ids, values):
        for s, v in zip(nest.flatten(self._state), nest.flatten(values)):
            s[ids] = v

    def add(self, ids, values):
        for s, v in zip(nest.flatten(self._state), nest.flatten(values)):
            s[ids] += v

    def clear(self, ids):
        for s in nest.flatten(self._state):
            # .zero_() doesn't work with tensor indexing?
            s[ids] = 0

    def __str__(self):
        return str(self._state)


def checkpoint(model, optimizer, scheduler, flags):
    if not flags.checkpoint:
        return

    logging.info("Saving checkpoint to %s", flags.checkpoint)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "flags": vars(flags),
        },
        flags.checkpoint,
    )


def make_train_model(flags, device=None):
    """Create the intended model according to options in `flags`"""
    model = SimpleNet(
        input_size=flags.observation_length,
        hidden_size=flags.hidden_size,
        num_actions=flags.num_actions,
        use_lstm=flags.use_lstm,
        n_train_jobs=get_n_jobs(flags, mode="train"),
        use_job_id_in_actor=flags.use_job_id_in_actor,
        use_job_id_in_critic=flags.use_job_id_in_critic,
    )
    return model if device is None else model.to(device)


def load_checkpoint(model, optimizer, scheduler, flags):
    # Return a boolean flag indicating whether a model checkpoint was loaded.
    if not flags.checkpoint or not os.path.exists(flags.checkpoint):
        return False

    state_dict = torch.load(flags.checkpoint)
    model.load_state_dict(state_dict["model_state_dict"])
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    scheduler.load_state_dict(state_dict["scheduler_state_dict"])

    logging.info("Loaded checkpoint from %s", flags.checkpoint)
    return True


def log(flags, _state={}, **fields):  # noqa: B008
    if "writer" not in _state:
        if not flags.logdir:
            _state["writer"] = None
            return

        path = os.path.join(flags.logdir, "logs.tsv")
        writeheader = not os.path.exists(path)
        fieldnames = list(fields.keys())

        _state["file"] = open(path, "a", buffering=1)  # Line buffering.
        _state["writer"] = csv.DictWriter(_state["file"], fieldnames, delimiter="\t")
        if writeheader:
            _state["writer"].writeheader()

    writer = _state["writer"]
    if writer is not None:
        writer.writerow(fields)


def learner_loop(flags, rank=0, barrier=None, gossip_buffer=None, stop_event=None):
    if flags.num_actors < flags.batch_size:
        logging.warn("Batch size is larger than number of actors.")
    assert (
        flags.batch_size % flags.inference_batch_size == 0
    ), "For now, inference_batch_size must divide batch_size"
    assert (
        flags.num_actors >= flags.inference_batch_size
    ), "Inference batch size must be <= number of actors"
    if flags.logdir:
        log_file_path = os.path.join(flags.logdir, "logs.tsv")
        logging.info(
            "%s logs to %s",
            "Appending" if os.path.exists(log_file_path) else "Writing",
            log_file_path,
        )
    else:
        logging.warn("--logdir not set. Not writing logs to file.")
    if not flags.cc_env_use_state_summary:
        raise NotImplementedError(
            "Setting cc_env_use_state_summary=False is currently not supported. The fetching "
            "of throughput and delay statistics through the states assumes that we are provided "
            "summary statistics, and would need to be updated to work without."
        )

    unroll_queue = queue.Queue(maxsize=1)
    log_queue = queue.Queue()

    # Inference model.
    model = make_train_model(flags)
    # Dummy (observation, reward, done)
    dummy_env_output = (
        np.zeros(flags.observation_length, dtype=np.float32),
        np.array(0, dtype=np.float32),
        np.array(True, dtype=np.bool),
    )
    dummy_env_output = nest.map(
        lambda a: torch.from_numpy(np.array(a)), dummy_env_output
    )

    with torch.no_grad():
        dummy_model_output, _ = model(
            last_actions=torch.zeros([1], dtype=torch.int64),
            env_outputs=nest.map(lambda t: t.unsqueeze(0), dummy_env_output),
            core_state=model.initial_state(1),
        )
        dummy_model_output = nest.map(lambda t: t.squeeze(0), dummy_model_output)

    model = model.to(device=flags.inference_device)

    # TODO: Decide if we really want that for simple tensors?
    actions = StructuredBuffer(torch.zeros([flags.num_actors], dtype=torch.int64))
    actor_run_ids = StructuredBuffer(torch.zeros([flags.num_actors], dtype=torch.int64))
    actor_infos = StructuredBuffer(
        dict(
            episode_step=torch.zeros([flags.num_actors], dtype=torch.int64),
            episode_return=torch.zeros([flags.num_actors]),
            cwnd_mean=torch.zeros([flags.num_actors]),
            delay_mean=torch.zeros([flags.num_actors]),
            throughput_mean=torch.zeros([flags.num_actors]),
            train_job_id=torch.zeros([flags.num_actors], dtype=torch.int64),
        )
    )

    # Agent states at the beginning of an unroll. Needs to be kept for learner.
    # A state is a tuple of two tensors of shape [num_actors, hidden_size + 1]):
    #   - Why two tensors? Because the LSTM cell's state contains both its output and its
    #     internal state.
    #   - Why +1? Because the reward is appended to the hidden layer being fed to the LSTM,
    #     and in `SimpleNet` the LSTM cell has the same output size as its input size.
    initial_states = model.initial_state(batch_size=flags.num_actors)
    first_agent_states = StructuredBuffer(initial_states)

    # Current agent states.
    agent_states = StructuredBuffer(copy.deepcopy(initial_states))

    rollouts = Rollouts(
        dict(
            last_actions=torch.zeros((), dtype=torch.int64),
            env_outputs=dummy_env_output,
            actor_outputs=dummy_model_output,
        ),
        unroll_length=flags.unroll_length,
        num_actors=flags.num_actors,
    )

    server = torchbeast.Server(flags.server_address, max_parallel_calls=4)

    def inference(actor_ids, run_ids, env_outputs):
        """
        Compute actions for a subset of actors, based on their observations.

        :param actor_ids: 1D tensor with the indices of actors whose actions
            are requested.
        :param run_ids: 1D tensor of same length as `actor_ids`, with the
            associated run IDs (used to deal with preemption / crashing: a
            fresh env is expected to provide a new run ID to be sure we do
            not accidentally re-use outdated data). Note that this mechanism
            is not actually needed by the current RL congestion control env.
        :param env_outputs: Tuple of observation (N x obs_dim), reward (N) and
            done (N) tensors, with N the size of `actor_ids`.
        """
        torch.set_grad_enabled(False)
        previous_run_ids = actor_run_ids.get(actor_ids)
        reset_indices = previous_run_ids != run_ids
        actor_run_ids.set(actor_ids, run_ids)

        actors_needing_reset = actor_ids[reset_indices]

        # Update new/restarted actors.
        # NB: this never happens because `runId` is always set to 0 in
        # `CongestionControlRPCEnv::makeCallRequest()``. This is working as
        # intended (calling `reset()` is not needed here).
        if actors_needing_reset.numel():
            logging.info("Actor ids needing reset: %s", actors_needing_reset.tolist())

            actor_infos.clear(actors_needing_reset)
            rollouts.reset(actors_needing_reset)
            actions.clear(actors_needing_reset)

            initial_agent_states = model.initial_state(actors_needing_reset.numel())
            first_agent_states.set(actors_needing_reset, initial_agent_states)
            agent_states.set(actors_needing_reset, initial_agent_states)

        obs, reward, done = env_outputs

        # Update logging stats at end of episode.
        done_ids = actor_ids[done]
        if done_ids.numel():
            # Do not log stats of zero-length episodes (typically these should
            # only happen on the very first episode, due to the `done` flag
            # being true without having any episode before).
            valid_done_ids = done_ids[actor_infos.get(done_ids)["episode_step"] > 0]
            log_queue.put((valid_done_ids, actor_infos.get(valid_done_ids)))
            # NB: `actor_infos.get()` returned a copy of the data, so it is ok
            # to clear it now.
            actor_infos.clear(done_ids)
            # Clear reward for agents that are done: it is meaningless as it is obtained with
            # the first observation, before the agent got a chance to take any action.
            reward[done] = 0.0
            # We only update the `train_job_id` field once (when an episode starts, which
            # is when `done` is True). This is because it remains the same throughout the
            # whole episode.
            extra_info = {"train_job_id": (obs[:, -1] * done).long()}
        else:
            extra_info = {}
        actor_infos.add(
            actor_ids,
            dict(
                episode_step=1,
                episode_return=reward,
                cwnd_mean=state.get_mean(obs, state.Field.CWND, dim=1)
                * flags.cc_env_norm_bytes
                / UDP_SEND_PACKET_LEN,
                delay_mean=state.get_mean(obs, state.Field.DELAY, dim=1)
                * flags.cc_env_norm_ms,
                throughput_mean=state.get_mean(obs, state.Field.THROUGHPUT, dim=1)
                * flags.cc_env_norm_bytes
                / 1024 ** 2,  # convert to Mbytes/s
                **extra_info,
            ),
        )

        last_actions = actions.get(actor_ids)
        prev_agent_states = agent_states.get(actor_ids)

        actor_outputs, new_agent_states = model(
            *nest.map(
                lambda t: t.to(flags.inference_device),
                (last_actions, env_outputs, prev_agent_states),
            )
        )
        actor_outputs, new_agent_states = nest.map(
            lambda t: t.cpu(), (actor_outputs, new_agent_states)
        )

        timestep = dict(
            last_actions=last_actions,
            env_outputs=env_outputs,
            actor_outputs=actor_outputs,
        )
        completed_ids, unrolls = rollouts.append(actor_ids, timestep)
        if completed_ids.numel():
            try:
                unroll_queue.put(
                    (completed_ids, unrolls, first_agent_states.get(completed_ids)),
                    timeout=5.0,
                )
            except queue.Closed:
                if server.running():
                    raise

            # Importantly, `first_agent_states` must contain the states *before* processing
            # the current batch of data, because in `rollouts` we always start from the last
            # item of the previous rollout (and we need the state before that one). This is
            # why this line must happen before the update to `agent_states` below.
            first_agent_states.set(completed_ids, agent_states.get(completed_ids))

        agent_states.set(actor_ids, new_agent_states)

        action = actor_outputs["action"]
        actions.set(actor_ids, action)
        return action

    server.bind("inference", inference, batch_size=flags.inference_batch_size)
    server.run()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        try:
            learn(
                model,
                executor,
                unroll_queue,
                log_queue,
                flags,
                rank,
                barrier,
                gossip_buffer,
                stop_event=stop_event,
            )
        except KeyboardInterrupt:
            print("Stopping ...")
        finally:
            unroll_queue.close()
            server.stop()
        # Need to shut down executor after queue is closed.


def learn(
    inference_model,
    executor,
    unroll_queue,
    log_queue,
    flags,
    rank=0,
    barrier=None,
    gossip_buffer=None,
    stop_event=None,
):
    assert flags.mode == "train"
    train_jobs = get_jobs(flags)

    model = make_train_model(flags, flags.learner_device)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    steps_per_epoch = flags.batch_size * flags.unroll_length

    if flags.learning_rate_decay == "linear":
        lr_lambda = (
            lambda epoch: 1
            - min(epoch * steps_per_epoch, flags.total_steps) / flags.total_steps
        )
    elif flags.learning_rate_decay == "none":
        lr_lambda = lambda epoch: 1.0
    else:
        raise NotImplementedError(flags.learning_rate_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    checkpoint_loaded = load_checkpoint(model, optimizer, scheduler, flags)
    inference_model.load_state_dict(model.state_dict())

    steps = scheduler.last_epoch * steps_per_epoch

    if steps >= flags.total_steps:
        # This should only happen if we loaded a checkpoint or explicitly asked
        # to run zero training step. Warn the user as no further training will
        # be performed.
        assert checkpoint_loaded or flags.total_steps == 0
        logging.warn(
            "Number of steps already performed (%d) is >= total "
            "number of steps (%d): *** NO FURTHER TRAINING WILL OCCUR ***",
            steps,
            flags.total_steps,
        )

    last_step = steps
    last_time = timeit.default_timer()

    current_time = last_time
    last_checkpoint_time = last_time
    tb_writer = (
        SummaryWriter(os.path.join(flags.logdir, "tensorboard"))
        if flags.logdir
        else None
    )

    def load_on_gpu():
        # TODO: Use CUDA streams?
        entries = 0
        next_batch = []

        while entries < flags.batch_size:
            # TODO: This isn't guaranteed to be exact if inference_batch_size
            # does not divide batch_size evenly.
            ids, *data = unroll_queue.get()
            next_batch.append(data)
            entries += ids.numel()

        # Batch.
        batch, initial_agent_state = nest.map_many(lambda d: torch.cat(d), *next_batch)

        # Make time major (excluding agent states).
        # After this step, tensors in `batch` are of shape (T + 1, N, ...) with
        # `T` the unroll length and `N` the number of actors in the batch.
        for t in nest.flatten(batch):
            t.transpose_(0, 1)

        if not flags.learner_device.startswith("cuda"):
            return nest.map(lambda t: t.contiguous(), (batch, initial_agent_state))
        return nest.map(
            lambda t: t.to(flags.learner_device, memory_format=torch.contiguous_format),
            (batch, initial_agent_state),
        )

    def log_target(steps, current_time, metrics_values):
        nonlocal last_step
        nonlocal last_time

        sps = (steps - last_step) / (current_time - last_time)

        if tb_writer is not None:
            for metric_name, metric_val in metrics_values.items():
                tb_writer.add_scalar(f"learner/{metric_name}", metric_val, steps)
            tb_writer.add_scalar("learner/sps", sps, steps)

        episode_returns = []

        for _ in range(log_queue.qsize()):
            ids, infos = log_queue.get()
            for (
                actor_id,
                episode_step,
                episode_return,
                cwnd_mean,
                delay_mean,
                throughput_mean,
                train_job_id,
            ) in zip(
                ids.tolist(),
                infos["episode_step"].tolist(),
                infos["episode_return"].tolist(),
                infos["cwnd_mean"].tolist(),
                infos["delay_mean"].tolist(),
                infos["throughput_mean"].tolist(),
                infos["train_job_id"].tolist(),
            ):
                episode_returns.append(episode_return)

                # These quantities must be averaged over all steps, not summed.
                cwnd_mean /= episode_step
                delay_mean /= episode_step
                throughput_mean /= episode_step

                # At this point, `train_job_id` is the index of the job in the
                # list of training jobs for this experiment. But for logging
                # purpose, we want to store the job ID as it would appear in
                # the `train_job_ids` option (i.e., its index in the list of
                # all jobs from `experiments.yml`). We do the conversion here.
                job_id = train_jobs[train_job_id]["job_id"]

                if tb_writer is not None:
                    tb_writer.add_scalar("actor/episode_steps", episode_step, steps)
                    tb_writer.add_scalar("actor/episode_return", episode_return, steps)
                    tb_writer.add_scalar("actor/cwnd_mean", cwnd_mean, steps)
                    tb_writer.add_scalar("actor/delay_mean", delay_mean, steps)
                    tb_writer.add_scalar(
                        "actor/throughput_mean", throughput_mean, steps
                    )

                log(
                    flags=flags,
                    step=steps,
                    episode_step=episode_step,
                    episode_return=episode_return,
                    actor_id=actor_id,
                    job_id=job_id,
                    sps=sps,
                    loss=metrics_values["loss/total"],
                    cwnd_mean=cwnd_mean,
                    delay_mean=delay_mean,
                    throughput_mean=throughput_mean,
                    timestep=time.time(),
                )

        # Log every 100 steps (roughly -- depends on `steps_per_epoch`).
        if steps // 100 > last_step // 100:
            if episode_returns:
                logging.info(
                    "Step %i @ %.1f SPS. Mean episode return: %f. "
                    "Episodes finished: %i. Loss: %f.",
                    steps,
                    sps,
                    sum(episode_returns) / len(episode_returns),
                    len(episode_returns),
                    metrics_values["loss/total"],
                )
            else:
                logging.info(
                    "Step %i @ %.1f SPS. Loss: %f.",
                    steps,
                    sps,
                    metrics_values["loss/total"],
                )

        last_step = steps
        last_time = current_time

    batch_future = executor.submit(load_on_gpu)
    log_future = executor.submit(lambda: None)

    # Synchronize GALA agents before starting training
    if barrier is not None:
        barrier.wait()
        logging.info("%s: barrier passed" % rank)

    reward_stats = None
    if flags.reward_normalization:
        all_job_ids = (
            range(get_n_jobs(flags))
            if flags.reward_normalization_stats_per_job
            else [0]
        )
        reward_stats = {
            job_id: {"initial_rewards": torch.zeros(0, device=flags.learner_device)}
            for job_id in all_job_ids
        }

    while steps < flags.total_steps:
        batch, initial_agent_state = batch_future.result()
        batch_future = executor.submit(load_on_gpu)

        learner_outputs, _ = model(
            batch["last_actions"],
            batch["env_outputs"],
            initial_agent_state,
            unroll=True,
        )

        metrics = compute_metrics(
            flags,
            learner_outputs,
            end_of_episode_bootstrap=flags.end_of_episode_bootstrap,
            reward_stats=reward_stats,
            **batch,
        )

        optimizer.zero_grad()
        metrics["loss/total"].backward()
        if flags.grad_norm_clipping:
            grad_norm = torch.as_tensor(
                nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
            )
            metrics["grad/norm"] = grad_norm
        optimizer.step()
        scheduler.step()

        # Local-Gossip in GALA mode
        if gossip_buffer is not None:
            gossip_buffer.write_message(rank, model)
            gossip_buffer.aggregate_message(rank, model)

        steps += steps_per_epoch
        metrics_values = {
            metric_name: metric_val.item()
            for metric_name, metric_val in metrics.items()
        }

        current_time = timeit.default_timer()

        if current_time - last_checkpoint_time > 10 * 60:  # Every 10 min.
            checkpoint(model, optimizer, scheduler, flags)
            last_checkpoint_time = timeit.default_timer()

        inference_model.load_state_dict(model.state_dict())

        log_future.result()
        log_future = executor.submit(log_target, steps, current_time, metrics_values)

    logging.info("Learning finished after %i steps", steps)
    checkpoint(model, optimizer, scheduler, flags)

    if stop_event is not None:
        logging.debug("Synching with main process to trigger env shutdown")
        stop_event.set()
        while stop_event.is_set():  # wait until main process clears event to continue
            time.sleep(0.1)
        logging.debug("Env shutdown completed -- ready to stop server")


def trace(flags):
    model = make_train_model(flags)
    model.eval()

    logging.info("Initializing weights from {} for tracing.".format(flags.checkpoint))
    device = torch.device("cpu")
    checkpoint = torch.load(flags.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    trace_model(flags, model)


def trace_model(flags, model):
    if not flags.traced_model:
        return

    model.eval()
    model = model.to(torch.device("cpu"))
    traced_model = torch.jit.trace(
        model,
        (
            torch.LongTensor(1),  # last_actions: [B]
            (
                torch.rand(1, flags.observation_length),  # observation [B, DIM]
                torch.rand(1),  # reward: [B]
                torch.BoolTensor(1),  # done: [B]
            ),
            model.initial_state(),  # core_state: [B, HIDDEN_DIM]
        ),
    )

    logging.info("Saving traced model to %s", flags.traced_model)
    traced_model.save(flags.traced_model)

    assert flags.traced_model.endswith(".pt"), flags.tracing
    flags_filename = flags.traced_model[:-3] + ".flags.pkl"
    logging.info("Saving flags to %s", flags_filename)
    with open(flags_filename, "wb") as f:
        # Dump with protocol 2 so that we can read the flags file in Python 2 in Pantheon.
        pickle.dump(vars(flags), f, 2)


def main(flags, rank=0, barrier=None, gossip_buffer=None, stop_event=None):
    try:
        return _main(
            flags=flags,
            rank=rank,
            barrier=barrier,
            gossip_buffer=gossip_buffer,
            stop_event=stop_event,
        )
    finally:  # make sure `stop_event` is set when the process exits
        if stop_event is not None:
            stop_event.set()


def _main(flags, rank=0, barrier=None, gossip_buffer=None, stop_event=None):
    torch.random.manual_seed(flags.seed)

    if flags.logdir:
        # Write meta.json file with some information on our setup.
        metadata = {
            "flags": OmegaConf.to_container(flags),
            "env": os.environ.copy(),
            "date_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            import git
        except ImportError:
            pass
        else:
            try:
                repo = git.Repo(search_parent_directories=True)
                metadata["git"] = {
                    "commit": repo.commit().hexsha,
                    "is_dirty": repo.is_dirty(),
                    "path": repo.git_dir,
                }
                if not repo.head.is_detached:
                    metadata["git"]["branch"] = repo.active_branch.name
            except git.InvalidGitRepositoryError:
                pass

        if "git" not in metadata:
            logging.warn("Couldn't determine git data.")

        with open(os.path.join(flags.logdir, "meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    if flags.mode == "train":
        learner_loop(flags, rank, barrier, gossip_buffer, stop_event)
    elif flags.mode == "trace":
        trace(flags)
    else:
        # Test mode unsupported in learner. We rely on "local" testing
        # by tracing the model and running it via C++ in mvfst without RPC.
        raise RuntimeError("Unsupported mode {}".format(flags.mode))

    if flags.logdir:
        # Write an empty "OK" flag to indicate success.
        with (Path(flags.logdir) / "OK").open("w"):
            pass
