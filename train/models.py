# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import logging
import operator

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

import nest


class SimpleNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_actions: int,
        use_lstm: bool = False,
        n_train_jobs: Optional[int] = None,
        use_job_id_in_actor: bool = False,
        use_job_id_in_critic: bool = False,
    ):
        super(SimpleNet, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        # If provided, `n_train_jobs` is the total number of jobs (= different environments)
        # this model is trained on. It is required to provide a one-hot vector of the
        # corresponding size as input to the actor and/or critic (depending on the flags
        # `use_job_id_in_{actor,critic}`).
        # Note that if there is only one training job, it is ignored since there is no
        # need to differentiate between multiple environments.
        self.use_job_id_in_network_input = False
        self.use_job_id_in_actor_head = False
        self.use_job_id_in_critic_head = False
        if use_job_id_in_actor or use_job_id_in_critic:
            assert n_train_jobs is not None and n_train_jobs >= 1, n_train_jobs
            self.n_train_jobs = 0 if n_train_jobs == 1 else n_train_jobs
            if self.n_train_jobs > 0:
                # If the job ID is provided as input to both actor and critic, then
                # its one-hot representation is given as input to the first layer.
                # Otherwise, it will be provided only to the actor or critic head.
                if use_job_id_in_actor and use_job_id_in_critic:
                    self.use_job_id_in_network_input = True
                else:
                    self.use_job_id_in_actor_head = use_job_id_in_actor
                    self.use_job_id_in_critic_head = use_job_id_in_critic
        else:
            self.n_train_jobs = 0  # not used

        # Feature extraction.
        # The first layer's input size is decreased by 1 because we remove the integer
        # representation of the job ID from the input.
        self.fc1 = nn.Linear(
            input_size - 1 + self.n_train_jobs * self.use_job_id_in_network_input,
            hidden_size,
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # FC output size + last reward.
        core_output_size = self.fc2.out_features + 1

        if use_lstm:
            self.core = nn.LSTMCell(core_output_size, core_output_size)

        self.policy = nn.Linear(
            core_output_size + self.n_train_jobs * self.use_job_id_in_actor_head,
            self.num_actions,
        )
        self.baseline = nn.Linear(
            core_output_size + self.n_train_jobs * self.use_job_id_in_critic_head,
            1,
        )

    def concat_to_job_id(self, tensor, job_id):
        """Concatenate a 2D tensor with the one-hot encoding associated to `job_id`"""
        one_hot = (
            F.one_hot(
                # We clamp the job ID to its target range. It is the responsibility
                # of the caller to ensure this does not cause problems downstream.
                # In particular this is useful for evaluation actors whose job ID
                # may be >= the number of training jobs.
                torch.clamp(job_id.flatten().long(), 0, self.n_train_jobs - 1),
                self.n_train_jobs,
            )
            .type(tensor.dtype)
            .to(tensor.device)
        )
        return torch.cat((tensor, one_hot), dim=1)

    def initial_state(self, batch_size=1):
        # Always return a tuple of two tensors so torch script type-checking
        # passes. It's sufficient for core state to be
        # Tuple[Tensor, Tensor] - the shapes don't matter.
        if self.use_lstm:
            core_hidden_size = self.core.hidden_size
        else:
            core_hidden_size = 0

        return tuple(torch.zeros(batch_size, core_hidden_size) for _ in range(2))

    def forward(self, last_actions, env_outputs, core_state, unroll=False):
        if not unroll:
            # [T=1, B, ...].
            env_outputs = nest.map(lambda t: t.unsqueeze(0), env_outputs)

        observation, reward, done = env_outputs

        T, B, *_ = observation.shape
        x = torch.flatten(observation, 0, 1)  # Merge time and batch.
        x = x.view(T * B, -1)

        # Separate the job ID from the rest of the observation.
        job_id = x[:, -1]
        x = x[:, 0:-1]

        if self.use_job_id_in_network_input:
            x = self.concat_to_job_id(x, job_id)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # reward = torch.clamp(reward, -1, 1).view(T * B, 1).float()
        reward = reward.view(T * B, 1).float()
        core_input = torch.cat([x, reward], dim=1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float()
            notdone.unsqueeze_(-1)  # [T, B, H=1] for broadcasting.

            for input_t, notdone_t in zip(core_input.unbind(), notdone.unbind()):
                # When `done` is True it means this is the first step in a new
                # episode => reset the internal state to zero.
                core_state = nest.map(notdone_t.mul, core_state)
                output_t, core_state = self.core(input_t, core_state)
                core_state = (output_t, core_state)  # nn.LSTMCell is a bit weird.
                core_output_list.append(output_t)  # [[B, H], [B, H], ...].
            core_output = torch.cat(core_output_list)  # [T * B, H].
        else:
            core_output = core_input

        actor_input = (
            self.concat_to_job_id(core_output, job_id)
            if self.use_job_id_in_actor_head
            else core_output
        )
        policy_logits = self.policy(actor_input)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

            critic_input = (
                self.concat_to_job_id(core_output, job_id)
                if self.use_job_id_in_critic_head
                else core_output
            )
            baseline = self.baseline(critic_input)

            baseline = baseline.view(T, B)

        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        action = action.view(T, B)

        if self.training:
            outputs = dict(
                action=action, policy_logits=policy_logits, baseline=baseline
            )
            if not unroll:
                outputs = nest.map(lambda t: t.squeeze(0), outputs)
            return outputs, core_state
        else:
            # In eval mode, we just return (action, core_state). PyTorch doesn't
            # support jit tracing output dicts.
            return action, core_state
