#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Main training script.
#
# Local run:
#   python3 -m train.train
# SLURM run:
#   python3 -m train.train hydra/launcher=submitit_slurm -m

import copy
import logging
import os
import shutil

from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp

from train import config, learner, pantheon_env, utils
from train.constants import CONF_ROOT, THIRD_PARTY_ROOT

utils.add_to_path(THIRD_PARTY_ROOT)

from gala.gpu_gossip_buffer import GossipBuffer
from gala.graph_manager import FullyConnectedGraph as Graph

logging.basicConfig(level=logging.INFO)

os.environ["OMP_NUM_THREADS"] = "1"

# Initialize config on import.
config.init_config()


def init_logdirs(flags):
    flags.logdir = os.path.join(flags.base_logdir, flags.mode)

    # Clean run for test mode
    if flags.mode != "train" and os.path.exists(flags.logdir):
        shutil.rmtree(flags.logdir)

    os.makedirs(flags.logdir, exist_ok=True)

    flags.checkpoint = os.path.join(flags.base_logdir, "checkpoint.tar")
    flags.traced_model = os.path.join(flags.base_logdir, "traced_model.pt")

    if flags.mode != "train" and "mvfst_rl" in pantheon_env.get_test_schemes(flags):
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
    model = learner.make_train_model(flags, device)

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


def train(flags):
    flags.mode = "train"
    flags.cc_env_mode = "remote"

    if torch.cuda.is_available():
        flags.learner_device = "cuda:0"
        flags.inference_device = "cuda:1"

    # For GALA
    proc_manager = mp.Manager()
    barrier = None
    shared_gossip_buffer = None

    # In GALA mode, start multiple replicas of the torchbeast-pantheon setup.
    num_agents = 1
    if flags.num_gala_agents > 1:
        num_agents = flags.num_gala_agents
        logging.info("In GALA mode, will start {} agents".format(num_agents))
        barrier = proc_manager.Barrier(num_agents)

        # Shared-gossip-buffer on GPU-0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        shared_gossip_buffer, _references = make_gossip_buffer(
            flags, num_agents, proc_manager, device
        )

    base_logdir = flags.base_logdir
    learner_proc = []
    pantheon_proc = []
    stop_event = []
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
        flags.server_address = "unix:{}".format(address)

        # Round-robin device assignment for GALA
        if num_agents > 1 and torch.cuda.is_available():
            flags.learner_device = "cuda:{}".format(rank % torch.cuda.device_count())
            flags.inference_device = "cuda:{}".format(rank % torch.cuda.device_count())

        logging.info(
            "Starting agent {}. Mode={}, logdir={}".format(
                rank, flags.mode, flags.logdir
            )
        )

        job_info_queue = mp.Queue()

        stop_event.append(mp.Event())
        learner_proc.append(
            mp.Process(
                target=learner.main,
                kwargs=dict(
                    flags=flags,
                    rank=rank,
                    barrier=barrier,
                    gossip_buffer=shared_gossip_buffer,
                    stop_event=stop_event[-1],
                    job_info_queue=job_info_queue,
                ),
                daemon=False,
            )
        )
        pantheon_proc.append(
            mp.Process(target=pantheon_env.main, args=(flags, job_info_queue), daemon=False)
        )
        learner_proc[rank].start()
        pantheon_proc[rank].start()

    # The shutdown sequence of a clean run is as follows:
    #   1. Wait until `stop_event` is set by the learner (=end of training notification)
    #   2. Kill the Pantheon process
    #   3. Clear `stop_event` to notify the learner it can exit (in particular, stop
    #      the RPC server).
    #   4. Wait until the learner process has exit
    # The motivation for this somewhat convoluted logic is that if we don't do #2 before
    # stopping the RPC server (in #3), then the Pantheon process will crash when the RPC
    # server is stopped, triggering meaningless error messages in the logs.
    for rank in range(num_agents):
        stop_event[rank].wait()
        logging.info(
            f"Stop event #{rank} set, will kill corresponding env (pid="
            f"{pantheon_proc[rank].pid})"
        )
        utils.kill_proc_tree(pantheon_proc[rank].pid)
        stop_event[rank].clear()
        learner_proc[rank].join()

    logging.info("Done training.")


def test(flags):
    flags.mode = "test"
    init_logdirs(flags)

    if "mvfst_rl" in pantheon_env.get_test_schemes(flags) and not os.path.exists(
        flags.traced_model
    ):
        logging.info("Missing traced model, tracing first")
        trace(copy.deepcopy(flags))

    flags.cc_env_mode = "local"

    logging.info("Starting local test, logdir={}".format(flags.logdir))
    job_info_queue = None
    pantheon_proc = mp.Process(target=pantheon_env.main, args=(flags, job_info_queue), daemon=False)
    pantheon_proc.start()
    pantheon_proc.join()
    logging.info("Done local test")


def trace(flags):
    flags.mode = "trace"
    init_logdirs(flags)

    logging.info("Tracing model from checkpoint {}".format(flags.checkpoint))
    learner_proc = mp.Process(target=learner.main, args=(flags,), daemon=False)
    learner_proc.start()
    learner_proc.join()
    assert learner_proc.exitcode == 0, "tracing failed"
    logging.info("Done tracing to {}".format(flags.traced_model))


def init(flags):
    """
    Initialization steps.
    """
    # Set log directory.
    if flags.base_logdir is None:
        flags.base_logdir = os.getcwd()
    else:
        # This should be an existing folder, typically one from an already
        # trained model.
        assert Path(flags.base_logdir).is_dir(), f"{flags.base_logdir} does not exist"

    # By default, Hydra changes the cwd to the experiment's directory (which we
    # use for logging purpose). In general it is best to avoid changing cwd, unless
    # there is a good reason for doing so. So we restore the original cwd.
    os.chdir(hydra.utils.get_original_cwd())

    # Compute `observation_length` from other settings.
    flags.observation_length = utils.get_observation_length(
        flags.cc_env_history_size, flags.num_actions
    )

    # Ensure the list of actions is properly set.
    if flags.cc_env_actions:
        assert len(flags.cc_env_actions) == flags.num_actions
    else:
        # Use default list of actions.
        flags.cc_env_actions = utils.get_actions(flags.num_actions)

    # Set the job IDs (independently for training and evaluation jobs).
    for jobs_type in ["train_jobs", "eval_jobs"]:
        all_job_ids = set()
        for idx, job in enumerate(flags[jobs_type].jobs.values()):
            if job.job_id is None:
                job.job_id = idx
            assert job.job_id not in all_job_ids, "duplicated job ID"
            all_job_ids.add(job.job_id)

    # Use "spawn" multi-processing mode as the default "fork" is not PyTorch-friendly.
    mp_start_method = mp.get_start_method(allow_none=True)
    if mp_start_method is None:
        mp.set_start_method("spawn")
    else:
        assert mp_start_method == "spawn"


@hydra.main(config_path=CONF_ROOT, config_name="config")
def main(flags) -> float:
    init(flags)
    mode = flags.mode
    logging.info("Mode={}".format(mode))

    if mode == "train":
        train(flags)
        if flags.test_after_train:
            test(flags)
    elif mode == "test":
        test(flags)
    elif mode == "trace":
        trace(flags)
    else:
        raise RuntimeError("Unknown mode {}".format(mode))

    logging.info(
        "All done! Checkpoint: {}, traced model: {}".format(
            flags.checkpoint, flags.traced_model
        )
    )

    # The return value is to be compatible with hyper-parameter optimization algorithms.
    return 0.0


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
