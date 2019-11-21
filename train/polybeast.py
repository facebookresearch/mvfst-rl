# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Run with OMP_NUM_THREADS=1.
#

import argparse
import collections
import functools
import logging
import operator
import os
import pickle
import sys
import threading
import time
import timeit
import traceback

import torch
from torch import nn
from torch.nn import functional as F

from train import common, utils
from train.constants import TORCHBEAST_ROOT

sys.path.append(TORCHBEAST_ROOT)

from core import file_writer
from core import vtrace

import nest

from libtorchbeast import actorpool


def add_args(parser):
    parser.add_argument("--xpid", default=None, help="Experiment id (default: None).")

    # Model output settings.
    parser.add_argument(
        "--checkpoint", default="checkpoint.tar", help="File to write checkpoints to."
    )

    # Model settings.
    parser.add_argument(
        "--observation_length",
        type=int,
        default=220,
        help="Length of the observation vector to be fed into the model.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden size in FC model."
    )
    parser.add_argument(
        "--num_actions",
        type=int,
        default=5,
        help="Number of actions output by the policy.",
    )

    # Training settings.
    parser.add_argument(
        "--address",
        default="unix:/tmp/rl_server_path",
        type=str,
        help="Address to bind ActorPoolServer to. Could be either "
        "<host>:<port> or Unix domain socket path unix:<path>.",
    )
    parser.add_argument(
        "--savedir",
        default="~/palaas/torchbeast",
        help="Root dir where experiment data will be saved.",
    )
    parser.add_argument(
        "--total_steps",
        default=1000000,
        type=int,
        metavar="T",
        help="Total environment steps to train for",
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, metavar="B", help="Learner batch size"
    )
    parser.add_argument(
        "--unroll_length",
        default=80,
        type=int,
        metavar="T",
        help="The unroll length (time dimension)",
    )
    parser.add_argument(
        "--num_learner_threads",
        default=2,
        type=int,
        metavar="N",
        help="Number of learner threads.",
    )
    parser.add_argument(
        "--num_inference_threads",
        default=2,
        type=int,
        metavar="N",
        help="Number of inference threads.",
    )
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument(
        "--use_lstm",
        type=utils.str2bool,
        default="true",
        help="Use LSTM in agent model.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Initial random seed for torch.random"
    )

    # GALA settings.
    parser.add_argument(
        "--num_gala_agents", default=1, type=int, help="If > 1, GALA mode is enabled."
    )
    parser.add_argument(
        "--num_gala_peers",
        type=int,
        default=1,
        help="Number of peers to communicate with in each iteration.",
    )
    parser.add_argument(
        "--sync_freq",
        type=int,
        default=0,
        help="Max amount of message staleness for local gossip.",
    )

    # Loss settings.
    parser.add_argument(
        "--entropy_cost", default=0.01, type=float, help="Entropy cost/multiplier."
    )
    parser.add_argument(
        "--baseline_cost", default=0.5, type=float, help="Baseline cost/multiplier."
    )
    parser.add_argument(
        "--discounting", default=0.99, type=float, help="Discounting factor."
    )
    parser.add_argument(
        "--reward_clipping",
        default="none",
        choices=["abs_one", "soft_asymmetric", "none"],
        help="Reward clipping.",
    )

    # Optimizer settings.
    parser.add_argument(
        "--learning_rate",
        default=0.00001,
        type=float,
        metavar="LR",
        help="Learning rate.",
    )
    parser.add_argument(
        "--alpha", default=0.99, type=float, help="RMSProp smoothing constant."
    )
    parser.add_argument("--momentum", default=0, type=float, help="RMSProp momentum.")
    parser.add_argument("--epsilon", default=0.01, type=float, help="RMSProp epsilon.")

    # Misc settings.
    parser.add_argument(
        "--write_profiler_trace",
        action="store_true",
        help="Collect and write a profiler trace " "for chrome://tracing/.",
    )


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


class Net(nn.Module):

    AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline")

    def __init__(self, observation_shape, hidden_size, num_actions, use_lstm=False):
        super(Net, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.use_lstm = use_lstm

        # Feature extraction.
        input_size = functools.reduce(operator.mul, observation_shape, 1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # FC output size + last reward.
        core_output_size = self.fc2.out_features + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, num_layers=1)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        # Always return a tuple of two tensors so torch script type-checking
        # passes. It's sufficient for core state to be
        # Tuple[Tensor, Tensor] - the shapes don't matter.
        if self.use_lstm:
            core_num_layers = self.core.num_layers
            core_hidden_size = self.core.hidden_size
        else:
            core_num_layers = 0
            core_hidden_size = 0

        return tuple(
            torch.zeros(core_num_layers, batch_size, core_hidden_size) for _ in range(2)
        )

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.view(T * B, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (1 - inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


def inference(inference_batcher, model, flags, lock=threading.Lock()):  # noqa: B008
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            frame, reward, done, *_ = batched_env_outputs
            frame = frame.to(flags.actor_device, non_blocking=True)
            reward = reward.to(flags.actor_device, non_blocking=True)
            done = done.to(flags.actor_device, non_blocking=True)
            agent_state = nest.map(
                lambda t: t.to(flags.actor_device, non_blocking=True), agent_state
            )
            with lock:
                outputs = model(
                    dict(frame=frame, reward=reward, done=done), agent_state
                )
            outputs = nest.map(lambda t: t.cpu(), outputs)
            batch.set_outputs(outputs)


EnvOutput = collections.namedtuple(
    "EnvOutput", "frame rewards done episode_step episode_return"
)
AgentOutput = Net.AgentOutput
Batch = collections.namedtuple("Batch", "env agent")


def learn(
    learner_queue,
    model,
    actor_model,
    optimizer,
    scheduler,
    stats,
    flags,
    plogger,
    rank=0,  # Rank of GALA agent
    gossip_buffer=None,  # Shared buffer for GALA gossip
    lock=threading.Lock(),  # noqa: B008
):
    for tensors in learner_queue:
        tensors = nest.map(lambda t: t.to(flags.learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        frame, reward, done, *_ = env_outputs

        lock.acquire()  # Only one thread learning at a time.
        learner_outputs, unused_state = model(
            dict(frame=frame, reward=reward, done=done), initial_agent_state
        )

        # Take final value function slice for bootstrapping.
        learner_outputs = AgentOutput._make(learner_outputs)
        bootstrap_value = learner_outputs.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(env_outputs.rewards, -1, 1)
        elif flags.reward_clipping == "soft_asymmetric":
            squeezed = torch.tanh(env_outputs.rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = (
                torch.where(env_outputs.rewards < 0, 0.3 * squeezed, squeezed) * 5.0
            )
        elif flags.reward_clipping == "none":
            clipped_rewards = env_outputs.rewards

        discounts = (1 - env_outputs.done).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits,
            actor_outputs.action,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs.policy_logits
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        scheduler.step()
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()

        # Local-Gossip in GALA mode
        if gossip_buffer is not None:
            gossip_buffer.write_message(rank, model)
            gossip_buffer.aggregate_message(rank, model)

        actor_model.load_state_dict(model.state_dict())

        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()

        stats["learner_queue_size"] = learner_queue.size()

        plogger.log(stats)

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        lock.release()


def train(flags, rank=0, barrier=None, device="cuda:0", gossip_buffer=None):
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )

    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.learner_device = torch.device(device)
        flags.actor_device = torch.device(device)
    else:
        logging.info("Not using CUDA.")
        flags.learner_device = torch.device("cpu")
        flags.actor_device = torch.device("cpu")

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static. We could make it dynamic, but that
    # requires a loss (and learning rate schedule) that's batch size
    # independent.
    learner_queue = actorpool.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
    )

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = actorpool.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    model = Net(
        observation_shape=flags.observation_shape,
        hidden_size=flags.hidden_size,
        num_actions=flags.num_actions,
        use_lstm=flags.use_lstm,
    )
    model = model.to(device=flags.learner_device)

    actor_model = Net(
        observation_shape=flags.observation_shape,
        hidden_size=flags.hidden_size,
        num_actions=flags.num_actions,
        use_lstm=flags.use_lstm,
    )
    actor_model.to(device=flags.actor_device)

    # The ActorPool that will accept connections from actor clients.
    actors = actorpool.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        server_address=flags.address,
        initial_agent_state=actor_model.initial_state(),
    )

    def run():
        try:
            actors.run()
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stats = {}

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                learner_queue,
                model,
                actor_model,
                optimizer,
                scheduler,
                stats,
                flags,
                plogger,
                rank,
                gossip_buffer,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(inference_batcher, actor_model, flags),
        )
        for i in range(flags.num_inference_threads)
    ]

    # Synchronize GALA agents before starting training
    if barrier is not None:
        barrier.wait()
        logging.info("%s: barrier passed" % rank)

    actorpool_thread.start()
    for t in learner_threads + inference_threads:
        t.start()

    def checkpoint():
        if flags.checkpoint:
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

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    try:
        last_checkpoint_time = timeit.default_timer()
        while True:
            start_time = timeit.default_timer()
            start_step = stats.get("step", 0)
            if start_step >= flags.total_steps:
                break
            time.sleep(5)
            end_step = stats.get("step", 0)

            if timeit.default_timer() - last_checkpoint_time > 10 * 60:
                # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timeit.default_timer()

            logging.info(
                "Step %i @ %.1f SPS. Inference batcher size: %i."
                " Learner queue size: %i."
                " Other stats: (%s)",
                end_step,
                (end_step - start_step) / (timeit.default_timer() - start_time),
                inference_batcher.size(),
                learner_queue.size(),
                ", ".join(
                    f"{key} = {format_value(value)}" for key, value in stats.items()
                ),
            )
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])
        checkpoint()

    # Done with learning. Let's stop all the ongoing work.
    inference_batcher.close()
    learner_queue.close()

    actors.stop()
    actorpool_thread.join()

    for t in learner_threads + inference_threads:
        t.join()

    # Trace and save the final model.
    trace_model(flags, model)


def trace(flags, **kwargs):
    model = Net(
        observation_shape=flags.observation_shape,
        hidden_size=flags.hidden_size,
        num_actions=flags.num_actions,
        use_lstm=flags.use_lstm,
    )
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
            dict(
                frame=torch.rand(*flags.observation_shape),
                reward=torch.rand(1, 1),
                done=torch.ByteTensor(1, 1),
            ),
            model.initial_state(),
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


def test(flags, **kwargs):
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA for testing.")
        flags.actor_device = torch.device("cuda:0")
    else:
        logging.info("Not using CUDA for testing.")
        flags.actor_device = torch.device("cpu")

    model = Net(
        observation_shape=flags.observation_shape,
        hidden_size=flags.hidden_size,
        num_actions=flags.num_actions,
        use_lstm=flags.use_lstm,
    )
    model.eval()

    logging.info("Initializing weights from {} for testing.".format(flags.checkpoint))
    checkpoint = torch.load(flags.checkpoint, map_location=flags.actor_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(flags.actor_device)

    inference_batcher = actorpool.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(inference_batcher, model, flags),
        )
        for i in range(flags.num_inference_threads)
    ]

    # Initialize ActorPool in test mode (without learner queue) for
    # RPC communication with the env and enqueueing steps in inference batcher.
    actors = actorpool.ActorPool(
        unroll_length=0,  # Unused in test mode
        learner_queue=None,  # Indicates test mode
        inference_batcher=inference_batcher,
        server_address=flags.address,
        initial_agent_state=model.initial_state(),
    )

    def run():
        try:
            actors.run()
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    actorpool_thread.start()
    for t in inference_threads:
        t.start()

    # Wait until interrupted
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass  # Close properly.

    logging.info("Testing finished.")
    inference_batcher.close()
    actors.stop()

    actorpool_thread.join()
    for t in inference_threads:
        t.join()


def main(flags, rank=0, barrier=None, device="cuda:0", gossip_buffer=None):
    torch.random.manual_seed(flags.seed)

    # We disable batching in learner as unroll lengths could different across
    # actors due to partial rollouts created by env resets.
    assert flags.batch_size == 1, "Batching in learner not supported currently"

    if flags.mode == "train" and flags.num_gala_agents > 1:
        # In GALA mode. Force single learner thread per GALA agent.
        flags.num_learner_threads = 1

    flags.observation_shape = (1, 1, flags.observation_length)
    kwargs = {
        "rank": rank,
        "barrier": barrier,
        "device": device,
        "gossip_buffer": gossip_buffer,
    }

    def error_fn(flags):
        raise RuntimeError("Unsupported mode {}".format(flags.mode))

    dispatch = {"train": train, "test": test, "trace": trace}
    run_fn = dispatch.get(flags.mode, error_fn)

    if flags.write_profiler_trace:
        logging.info("Running with profiler.")
        with torch.autograd.profiler.profile() as prof:
            run_fn(flags, **kwargs)
        filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
        logging.info("Writing profiler trace to '%s.gz'", filename)
        prof.export_chrome_trace(filename)
        os.system("gzip %s" % filename)
    else:
        run_fn(flags, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")
    common.add_args(parser)
    add_args(parser)
    flags = parser.parse_args()
    main(flags)
