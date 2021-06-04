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
        task_obs_size: int,
        num_actions: int,
        use_lstm: bool = False,
        use_reward: bool = True,
        use_task_obs_in_actor: bool = False,
        use_task_obs_in_critic: bool = False,
    ):
        super(SimpleNet, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.use_reward = use_reward

        self.use_task_obs_in_network_input = False
        self.use_task_obs_in_actor_head = False
        self.use_task_obs_in_critic_head = False
        # If the task observation is provided as input to both actor and critic,
        # then it is fed as input to the first layer.
        # Otherwise, it will be provided only to the actor or critic head.
        if use_task_obs_in_actor and use_task_obs_in_critic:
            self.use_task_obs_in_network_input = True
        else:
            self.use_task_obs_in_actor_head = use_task_obs_in_actor
            self.use_task_obs_in_critic_head = use_task_obs_in_critic

        # Feature extraction.
        self.fc1 = nn.Linear(
            input_size + task_obs_size * self.use_task_obs_in_network_input,
            hidden_size,
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # FC output size + (optionally) last reward.
        core_output_size = self.fc2.out_features + int(self.use_reward)

        if use_lstm:
            self.core = nn.LSTMCell(core_output_size, core_output_size)

        self.policy = nn.Linear(
            core_output_size + task_obs_size * self.use_task_obs_in_actor_head,
            self.num_actions,
        )
        self.baseline = nn.Linear(
            core_output_size + task_obs_size * self.use_task_obs_in_critic_head,
            1,
        )

    def concat_to_task_obs(self, tensor, task_obs):
        """Concatenate a 2D tensor with the task observation"""
        return torch.cat((tensor, task_obs.type(tensor.dtype).to(tensor.device)), dim=1)

    def initial_state(self, batch_size=1):
        # Always return a tuple of two tensors so torch script type-checking
        # passes. It's sufficient for core state to be
        # Tuple[Tensor, Tensor] - the shapes don't matter.
        if self.use_lstm:
            core_hidden_size = self.core.hidden_size
        else:
            core_hidden_size = 0

        return tuple(torch.zeros(batch_size, core_hidden_size) for _ in range(2))

    def forward(self, last_actions, env_outputs, task_obs, core_state, unroll=False):
        if not unroll:
            # [T=1, B, ...].
            task_obs, env_outputs = nest.map(lambda t: t.unsqueeze(0), (task_obs, env_outputs))

        observation, reward, done = env_outputs

        T, B, *_ = observation.shape
        x = torch.flatten(observation, 0, 1)  # Merge time and batch.
        x = x.view(T * B, -1)
        task_obs = task_obs.view(T * B, -1)

        if self.use_task_obs_in_network_input:
            x = self.concat_to_task_obs(x, task_obs)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.use_reward:
            # reward = torch.clamp(reward, -1, 1).view(T * B, 1).float()
            reward = reward.view(T * B, 1).float()
            core_input = torch.cat([x, reward], dim=1)
        else:
            core_input = x

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
            self.concat_to_task_obs(core_output, task_obs)
            if self.use_task_obs_in_actor_head
            else core_output
        )
        policy_logits = self.policy(actor_input)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

            critic_input = (
                self.concat_to_task_obs(core_output, task_obs)
                if self.use_task_obs_in_critic_head
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
