#!/usr/bin/env python3

import os
from os import path
import argparse
import logging
import threading
import time
import subprocess

import numpy as np
import sys

from constants import SRC_DIR, PANTHEON_ROOT

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Pantheon Environment Server')

parser.add_argument('--start_port', default=60000, type=int, metavar='P',
                    help='Server port for first environment.')
parser.add_argument('-N', '--num_env', default=4, type=int, metavar='N',
                    help='Number of environment servers.')


src_path = path.join(PANTHEON_ROOT, 'src/experiments/test.py')


def get_cmd(port):
    extra_sender_args = ' '.join([
        '--cc_env_mode=train',
        '--cc_env_port={}'.format(port),
    ])
    cmd = [
        src_path,
        'local',
        '--schemes', 'mvfst_rl',
        '--extra_sender_args', '"{}"'.format(extra_sender_args),
    ]
    return cmd


if __name__ == "__main__":
    flags = parser.parse_args()

    # $PATH override to put python2 first for Pantheon
    result = subprocess.run(
        ['dirname $(which python2)'],
        shell=True,
        stdout=subprocess.PIPE,
    )
    python2_path = result.stdout.decode('utf-8')
    logging.info('Located python2 in {}'.format(python2_path))

    pantheon_env = os.environ.copy()
    pantheon_env["PATH"] = python2_path + pantheon_env["PATH"]

    logging.info('Starting {} Pantheon env instances'.format(flags.num_env))

    processes = []
    for i in range(flags.num_env):
        cmd = get_cmd(flags.start_port + i)
        logging.info('Launch cmd: {}'.format(' '.join(cmd)))
        p = subprocess.Popen(cmd, env=pantheon_env)
        processes.append(p)

    for p in processes:
        p.wait()
