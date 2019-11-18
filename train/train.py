#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Run as python3 -m train.train

import argparse
import copy
import logging
import torch
import torch.multiprocessing as mp
import os
import shutil
import sys

from train import polybeast, pantheon_env, common, utils
from train.constants import THIRD_PARTY_ROOT

sys.path.append(THIRD_PARTY_ROOT)

from gala.gpu_gossip_buffer import GossipBuffer
from gala.graph_manager import FullyConnectedGraph as Graph

logging.basicConfig(level=logging.INFO)

os.environ["OMP_NUM_THREADS"] = "1"


def get_parser():
    parser = argparse.ArgumentParser()
    common.add_args(parser)

    polybeast_parser = parser.add_argument_group("polybeast")
    polybeast.add_args(polybeast_parser)

    pantheon_parser = parser.add_argument_group("pantheon_env")
    pantheon_env.add_args(pantheon_parser)

    parser.add_argument("--base_logdir", type=str, default="logs")

    return parser


def init_logdirs(flags):
    flags.logdir = os.path.join(flags.base_logdir, flags.mode)
    flags.savedir = os.path.join(flags.logdir, "torchbeast")

    # Clean run for test mode
    if flags.mode != "train" and os.path.exists(flags.logdir):
        shutil.rmtree(flags.logdir)

    os.makedirs(flags.logdir, exist_ok=True)
    os.makedirs(flags.savedir, exist_ok=True)

    flags.checkpoint = os.path.join(flags.base_logdir, "checkpoint.tar")
    flags.traced_model = os.path.join(flags.base_logdir, "traced_model.pt")

    if flags.mode != "train":
        assert os.path.exists(
            flags.checkpoint
        ), "Checkpoint {} missing in {} mode".format(flags.checkpoint, flags.mode)


def make_gossip_buffer(flags, num_agents, mng, device):
    """
    Shared gossip buffer for GALA mode.
    """
    if num_agents <= 1:
        return None, None

    # Make topology
    topology = []
    for rank in range(num_agents):
        graph = Graph(rank, num_agents, peers_per_itr=flags.num_gala_peers)
        topology.append(graph)

    # Initialize parameter buffer
    flags.observation_shape = (1, 1, flags.observation_length)
    model = polybeast.Net(
        observation_shape=flags.observation_shape,
        hidden_size=flags.hidden_size,
        num_actions=flags.num_actions,
        use_lstm=flags.use_lstm,
    )
    model.to(device)

    # Keep track of local iterations since learner's last sync
    sync_list = mng.list([0 for _ in range(num_agents)])
    # Used to ensure proc-safe access to agents' message-buffers
    buffer_locks = mng.list([mng.Lock() for _ in range(num_agents)])
    # Used to signal between processes that message was read
    read_events = mng.list(
        [mng.list([mng.Event() for _ in range(num_agents)]) for _ in range(num_agents)]
    )
    # Used to signal between processes that message was written
    write_events = mng.list(
        [mng.list([mng.Event() for _ in range(num_agents)]) for _ in range(num_agents)]
    )

    # Need to maintain a reference to all objects in main processes
    _references = [topology, model, buffer_locks, read_events, write_events, sync_list]
    gossip_buffer = GossipBuffer(
        topology,
        model,
        buffer_locks,
        read_events,
        write_events,
        sync_list,
        sync_freq=flags.sync_freq,
    )
    return gossip_buffer, _references


def run_remote(flags, train=True):
    flags.mode = "train" if train else "test"
    flags.disable_cuda = not train
    flags.cc_env_mode = "remote"

    proc_manager = mp.Manager()
    barrier = None
    shared_gossip_buffer = None
    cuda = not flags.disable_cuda and torch.cuda.is_available()

    num_agents = 1
    if train and flags.num_gala_agents > 1:
        # In GALA mode. Start multiple replicas of the polybeast-pantheon setup.
        num_agents = flags.num_gala_agents
        logging.info("In GALA mode, will start {} agents".format(num_agents))
        barrier = proc_manager.Barrier(num_agents)

        # Shared-gossip-buffer on GPU-0
        device = torch.device("cuda:0" if cuda else "cpu")
        shared_gossip_buffer, _references = make_gossip_buffer(
            flags, num_agents, proc_manager, device
        )

    base_logdir = flags.base_logdir
    polybeast_proc = []
    pantheon_proc = []
    for rank in range(num_agents):
        flags.base_logdir = (
            os.path.join(base_logdir, "gala_{}".format(rank))
            if num_agents > 1
            else base_logdir
        )
        init_logdirs(flags)

        # Unix domain socket path for RL server address, one per GALA agent.
        address = "/tmp/rl_server_path_{}".format(rank)
        try:
            os.remove(address)
        except OSError:
            pass
        flags.address = "unix:{}".format(address)
        flags.server_address = flags.address

        # Round-robin device assignment
        device = "cuda:{}".format(rank % torch.cuda.device_count()) if cuda else "cpu"

        logging.info(
            "Starting agent {} on device {}. Mode={}, logdir={}".format(
                rank, device, flags.mode, flags.logdir
            )
        )
        polybeast_proc.append(
            mp.Process(
                target=polybeast.main,
                args=(flags, rank, barrier, device, shared_gossip_buffer),
                daemon=False,
            )
        )
        pantheon_proc.append(
            mp.Process(target=pantheon_env.main, args=(flags,), daemon=False)
        )
        polybeast_proc[rank].start()
        pantheon_proc[rank].start()

    if train:
        # Training is driven by polybeast. Wait until it returns and then
        # kill pantheon_env.
        for rank in range(num_agents):
            polybeast_proc[rank].join()
            pantheon_proc[rank].kill()
    else:
        # Testing is driven by pantheon_env. Wait for it to join and then
        # kill polybeast.
        for rank in range(num_agents):
            pantheon_proc[rank].join()
            polybeast_proc[rank].kill()

    logging.info("Done {}".format(flags.mode))


def test_local(flags):
    flags.mode = "test"
    init_logdirs(flags)

    if not os.path.exists(flags.traced_model):
        logging.info("Missing traced model, tracing first")
        trace(copy.deepcopy(flags))

    flags.cc_env_mode = "local"

    logging.info("Starting local test, logdir={}".format(flags.logdir))
    pantheon_proc = mp.Process(target=pantheon_env.main, args=(flags,), daemon=False)
    pantheon_proc.start()
    pantheon_proc.join()
    logging.info("Done local test")


def trace(flags):
    flags.mode = "trace"
    init_logdirs(flags)

    logging.info("Tracing model from checkpoint {}".format(flags.checkpoint))
    polybeast_proc = mp.Process(target=polybeast.main, args=(flags,), daemon=False)
    polybeast_proc.start()
    polybeast_proc.join()
    logging.info("Done tracing to {}".format(flags.traced_model))


def main(flags):
    mode = flags.mode
    logging.info("Mode={}".format(mode))

    if mode == "train":
        # Train, trace, and then test
        run_remote(flags, train=True)
        trace(flags)
        run_remote(flags, train=False)
    elif mode == "test":
        # Only remote test
        run_remote(flags, train=False)
    elif mode == "test_local":
        # Only local test
        test_local(flags)
    elif mode == "trace":
        trace(flags)
    else:
        raise RuntimeError("Unknown mode {}".format(mode))

    logging.info(
        "All done! Checkpoint: {}, traced model: {}".format(
            flags.checkpoint, flags.traced_model
        )
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = get_parser()
    flags = parser.parse_args()
    main(flags)
