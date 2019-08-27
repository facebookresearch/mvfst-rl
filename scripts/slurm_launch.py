#!/usr/bin/env python3

import argparse
import datetime
import itertools
import logging
import os

import submitit

from train import train

logging.basicConfig(level=logging.INFO)

os.environ["OMP_NUM_THREADS"] = "1"


def add_args(parser):
    parser.add_argument("--local", default=False, action="store_true")


def main(flags):
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    logdir = "/checkpoint/{}/mvrlfst/{}".format(os.environ["USER"], now)
    os.makedirs(logdir, exist_ok=True)
    logging.info("Logdir: {}".format(logdir))

    train_parser = train.get_parser()
    train_flags = train_parser.parse_args(["--base_logdir={}".format(logdir)])

    if flags.local:
        executor = submitit.LocalExecutor(folder=logdir)
    else:
        executor = submitit.SlurmExecutor(folder=logdir)
    executor.update_parameters(
        partition="dev",
        time=600,
        nodes=1,
        ntasks_per_node=1,
        job_name="mvrlfst",
        num_gpus=2,
        cpus_per_task=40,
        mem="64GB",
    )
    job = executor.submit(train.main, train_flags)
    logging.info("Submitted job {}".format(job.job_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    flags = parser.parse_args()
    main(flags)
