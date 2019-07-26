#!/usr/bin/env python3

import os
from os import path
import argparse
import threading
import time
import subprocess
import shlex

import numpy as np
import sys

from constants import SRC_DIR, PANTHEON_ROOT
from helpers import utils

parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument('--start_port', default=60000, type=int, metavar='P',
                    help='Server port for first environment.')
parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')

test_path = path.join(PANTHEON_ROOT, 'src/experiments/test.py')
logs_path = path.join(SRC_DIR, 'train/logs')
def run_emu(flags):
    sys.stderr.write('----- Running emulation experiments -----\n')

    cfg = utils.expt_cfg['emu']
    matrix = utils.expand_matrix(cfg['matrix'])

    # create a queue of jobs
    job_queue = []
    for mat_dict in matrix:
        for job_cfg in cfg['jobs']:
            cmd_tmpl = job_cfg['command']

            # 1. expand macros
            cmd_tmpl = utils.safe_format(cmd_tmpl, cfg['macros'])
            # 2. expand variables in mat_dict
            cmd_tmpl = utils.safe_format(cmd_tmpl, mat_dict)
            # 3. expand meta
            cmd_tmpl = utils.safe_format(cmd_tmpl, utils.meta)
            cmd_tmpl = utils.safe_format(cmd_tmpl, {'src_dir' : SRC_DIR,
                           'data_dir': path.join(logs_path, 'sc_%d' % job_cfg['scenario'])})

            job_queue.append((job_cfg, cmd_tmpl))

    processes = []
    n = len(job_queue)
    for i in range(flags.num_servers):
        job_cfg, cmd = job_queue[i % n]
        with open(path.join(SRC_DIR, "sc_%d.log" % job_cfg['scenario']), 'w') as log_f:
            p = subprocess.Popen(get_cmd(cmd, flags.start_port + i), stdout=log_f, stderr=log_f)
    for p in processes:
        p.wait()

def get_cmd(cmd, port):
    cmd = shlex.split(cmd) + ['--extra_sender_args',
                             '--cc_env_port=%d --cc_env_mode=train' % port]
    cmd[0] = path.abspath(cmd[0])
    print(cmd)
    return cmd

if __name__ == "__main__":
    flags = parser.parse_args()
    run_emu(flags)
