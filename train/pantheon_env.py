#!/usr/bin/env python3

import os
from os import path
import argparse
import logging
import subprocess
import utils
import shlex

import sys

from constants import SRC_DIR, PANTHEON_ROOT

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Pantheon Environment Server')

parser.add_argument('--start_port', default=60000, type=int, metavar='P',
                    help='Server port for first environment.')
parser.add_argument('-N', '--num_env', default=4, type=int, metavar='N',
                    help='Number of environment servers.')


src_path = path.join(PANTHEON_ROOT, 'src/experiments/test.py')
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
            data_dir = path.join(logs_path, 'sc_%d' % job_cfg['scenario'])
            cmd_tmpl = utils.safe_format(cmd_tmpl,
                                         {'data_dir': data_dir})

            job_queue.append((job_cfg, cmd_tmpl))

    processes = []
    n = len(job_queue)
    for i in range(flags.num_env):
        job_cfg, cmd = job_queue[i % n]
        log_file_name = path.join(SRC_DIR, "sc_%d.log" % job_cfg['scenario'])
        with open(log_file_name, 'w') as log_f:
            cmd_to_process = get_cmd(cmd, flags.start_port + i)
            logging.info('Launch cmd: {}'.format(' '.join(cmd_to_process)))
            p = subprocess.Popen(cmd_to_process,
                                 stdout=log_f, stderr=log_f)
    for p in processes:
        p.wait()


def get_cmd(cmd, port):
    extra_sender_args = [
        '--cc_env_mode=train',
        '--cc_env_port={}'.format(port),
    ]
    cmd = shlex.split(cmd) + [
        '--extra_sender_args', ' '.join(extra_sender_args),
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
    python2_path = result.stdout.decode('utf-8').strip()
    logging.info('Located python2 in {}'.format(python2_path))

    pantheon_env = os.environ.copy()
    pantheon_env["PATH"] = ':'.join([python2_path, pantheon_env["PATH"]])

    logging.info('Starting {} Pantheon env instances'.format(flags.num_env))

    run_emu(flags)
