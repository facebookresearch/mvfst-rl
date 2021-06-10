# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Iterable, List, Optional

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
        activations: Iterable[str] = ("relu", "relu"),
        use_lstm: bool = False,
        use_reward: bool = True,
        use_task_obs_in_actor: bool = False,
        use_task_obs_in_critic: bool = False,
        # Whether the actor / critic heads are present. At least one must be present,
        # and if only one is present then only the corresponding `use_task_obs_in_XYZ`
        # flag is used.
        actor_head: bool = True,
        critic_head: bool = True,
        init_policy_logits_biases: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.use_reward = use_reward
        self.actor_head = actor_head
        self.critic_head = critic_head
        assert self.actor_head or self.critic_head

        # Decide where to plug the task observation.
        self.use_task_obs_in_network_input = False
        self.use_task_obs_in_actor_head = False
        self.use_task_obs_in_critic_head = False
        if (
            # Both actor and critic need it.
            (use_task_obs_in_actor and use_task_obs_in_critic)
            # Actor needs it, and this model only computes the actor output.
            or (use_task_obs_in_actor and not critic_head)
            # Critic needs it, and this model only computes the critic output.
            or (use_task_obs_in_critic and not actor_head)
        ):
            # Then we can add the task observation directly as input to the network.
            self.use_task_obs_in_network_input = True
        else:
            # Otherwise plug it only as input to (at most) one of the heads.
            self.use_task_obs_in_actor_head = use_task_obs_in_actor
            self.use_task_obs_in_critic_head = use_task_obs_in_critic

        # Feature extraction.
        # The length of `activations` also defines the number of fully connected layers.
        assert len(activations) >= 1
        self.fc_layers = nn.ModuleList(
            modules=[
                nn.Linear(
                    input_size + task_obs_size * self.use_task_obs_in_network_input,
                    hidden_size,
                )
            ]
            + [nn.Linear(hidden_size, hidden_size) for _ in range(len(activations) - 1)]
        )

        # Prepare list of activation functions.
        self.activations = []
        for act in activations:
            if act == "linear":
                self.activations.append(None)
            elif act == "relu":
                self.activations.append(F.relu)
            elif act == "tanh":
                self.activations.append(torch.tanh)
            else:
                raise NotImplementedError(act)

        # FC output size + (optionally) last reward.
        core_output_size = self.fc_layers[-1].out_features + int(self.use_reward)

        if use_lstm:
            self.core = nn.LSTMCell(core_output_size, core_output_size)

        if self.actor_head:
            self.policy = nn.Linear(
                core_output_size + task_obs_size * self.use_task_obs_in_actor_head,
                self.num_actions,
            )
            if init_policy_logits_biases:
                assert len(init_policy_logits_biases) == self.num_actions
                with torch.no_grad():
                    self.policy.bias[:] = torch.tensor(init_policy_logits_biases).type(
                        self.policy.bias.dtype
                    )

        if self.critic_head:
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
            task_obs, env_outputs = nest.map(
                lambda t: t.unsqueeze(0), (task_obs, env_outputs)
            )

        observation, reward, done = env_outputs

        T, B, *_ = observation.shape
        x = torch.flatten(observation, 0, 1)  # Merge time and batch.
        x = x.view(T * B, -1)
        task_obs = task_obs.view(T * B, -1)

        if self.use_task_obs_in_network_input:
            x = self.concat_to_task_obs(x, task_obs)

        for layer, activation in zip(self.fc_layers, self.activations):
            x = layer(x)
            if activation is not None:
                x = activation(x)

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

        if self.actor_head:
            actor_input = (
                self.concat_to_task_obs(core_output, task_obs)
                if self.use_task_obs_in_actor_head
                else core_output
            )
            policy_logits = self.policy(actor_input)

            if self.training:
                action = torch.multinomial(
                    F.softmax(policy_logits, dim=1), num_samples=1
                )
            else:
                # Don't sample when testing.
                action = torch.argmax(policy_logits, dim=1)

            policy_logits = policy_logits.view(T, B, self.num_actions)
            action = action.view(T, B)

        if self.training and self.critic_head:
            critic_input = (
                self.concat_to_task_obs(core_output, task_obs)
                if self.use_task_obs_in_critic_head
                else core_output
            )
            baseline = self.baseline(critic_input)
            baseline = baseline.view(T, B)

        if self.training:
            outputs = {}
            if self.actor_head:
                outputs["action"] = action
                outputs["policy_logits"] = policy_logits
            if self.critic_head:
                outputs["baseline"] = baseline
            if not unroll:
                outputs = nest.map(lambda t: t.squeeze(0), outputs)
            return outputs, core_state

        else:
            # A critic alone cannot be used in eval mode.
            assert self.actor_head
            # In eval mode, we just return (action, core_state). PyTorch doesn't
            # support jit tracing output dicts.
            return action, core_state


class SeparateActorCritic(nn.Module):
    """Used to avoid sharing parameters between actor and critic"""

    def __init__(self, **kw):
        super().__init__()
        self.actor = SimpleNet(**kw, actor_head=True, critic_head=False)
        self.critic = SimpleNet(**kw, actor_head=False, critic_head=True)

    def initial_state(self, *args, **kw):
        return (
            self.actor.initial_state(*args, **kw),
            self.critic.initial_state(*args, **kw),
        )

    def forward(self, last_actions, env_outputs, task_obs, core_state, unroll=False):
        actor_core_state, critic_core_state = core_state

        actor_outputs, actor_core_state = self.actor(
            last_actions=last_actions,
            env_outputs=env_outputs,
            task_obs=task_obs,
            core_state=actor_core_state,
            unroll=unroll,
        )

        if self.training:
            critic_outputs, critic_core_state = self.critic(
                last_actions=last_actions,
                env_outputs=env_outputs,
                task_obs=task_obs,
                core_state=critic_core_state,
                unroll=unroll,
            )
            outputs = actor_outputs.copy()
            outputs.update(critic_outputs)
            return outputs, (actor_core_state, critic_core_state)

        else:
            return actor_outputs, (actor_core_state, critic_core_state)
