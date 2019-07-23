#!/usr/bin/env python3

import os
from os import path
import argparse
import multiprocessing as mp
import threading
import time
import subprocess

import numpy as np
import sys


parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument('--start_port', default=60000, type=int, metavar='P',
                    help='Server port for first environment.')
parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')

src_dir = path.abspath(path.dirname(__file__))
test_path = path.join(src_dir, '_build', 'deps',
                    'pantheon', 'src', 'experiments', 'test.py')


def create_env(port, lock=threading.Lock()):
    cmd = [test_path, 'local', '--schemes', 'mvfst_rl', '--extra_sender_args',
                    '--cc_env_port=%d --cc_env_mode=train' % port]
    return subprocess.call(cmd)

if __name__ == "__main__":
    flags = parser.parse_args()

    processes = []
    for i in range(flags.num_servers):
        p = mp.Process(
            target=create_env, args=((flags.start_port + i),), daemon=True
        )
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
