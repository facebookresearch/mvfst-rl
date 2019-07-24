#!/usr/bin/env python3

import os
from os import path
import argparse
import threading
import time
import subprocess

import numpy as np
import sys

from constants import SRC_DIR, PANTHEON_ROOT


parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument('--start_port', default=60000, type=int, metavar='P',
                    help='Server port for first environment.')
parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')

test_path = path.join(PANTHEON_ROOT, 'src/experiments/test.py')


def get_cmd(port):
    cmd = [test_path, 'local', '--schemes', 'mvfst_rl',
                    '--extra_sender_args', '--cc_env_port=%d --cc_env_mode=train' % port]
    return cmd

if __name__ == "__main__":
    flags = parser.parse_args()

    processes = []
    for i in range(flags.num_servers):
        p = subprocess.Popen(get_cmd(flags.start_port + i))
        processes.append(p)

    for p in processes:
        p.wait()
