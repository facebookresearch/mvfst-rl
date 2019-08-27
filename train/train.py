#!/usr/bin/env python3

# Run as python3 -m train.train

import argparse
import logging
import multiprocessing as mp
import os

from train import polybeast, pantheon_env, common

logging.basicConfig(level=logging.INFO)

os.environ["OMP_NUM_THREADS"] = "1"


def get_parser():
    parser = argparse.ArgumentParser()
    common.add_args(parser)

    polybeast_parser = parser.add_argument_group("polybeast")
    polybeast.add_args(polybeast_parser)

    pantheon_parser = parser.add_argument_group("pantheon_env")
    pantheon_env.add_args(pantheon_parser)

    parser.add_argument(
        "--test_only",
        default=False,
        action="store_true",
        help="If set, only test is run",
    )
    parser.add_argument("--base_logdir", type=str, default="logs")

    return parser


def run(flags, train=True):
    flags.mode = "train" if train else "test"

    flags.logdir = os.path.join(flags.base_logdir, flags.mode)
    flags.savedir = os.path.join(flags.logdir, "torchbeast")
    os.makedirs(flags.logdir, exist_ok=True)
    os.makedirs(flags.savedir, exist_ok=True)

    flags.checkpoint = os.path.join(flags.base_logdir, "checkpoint.tar")

    # Unix domain socket path for RL server address
    address = "/tmp/rl_server_path"
    if os.path.exists(address):
        os.remove(address)
    flags.address = "unix:{}".format(address)

    flags.disable_cuda = not train

    logging.info("Starting {}, logdir={}".format(flags.mode, flags.logdir))
    polybeast_proc = mp.Process(target=polybeast.main, args=(flags,))
    pantheon_proc = mp.Process(target=pantheon_env.main, args=(flags,))
    polybeast_proc.start()
    pantheon_proc.start()

    if train:
        # Training is driven by polybeast. Wait until it returns and then
        # kill pantheon_env.
        polybeast_proc.join()
        pantheon_proc.kill()
    else:
        # Testing is driven by pantheon_env. Wait for it to join and then
        # kill polybeast.
        pantheon_proc.join()
        polybeast_proc.kill()

    logging.info("Done {}".format(flags.mode))


def main(flags):
    if not flags.test_only:
        run(flags, train=True)
    run(flags, train=False)

    logging.info("All done! Model: {}".format(flags.checkpoint))


if __name__ == "__main__":
    parser = get_parser()
    flags = parser.parse_args()
    main(flags)
