#!/usr/bin/env python3

import os
from os import path
import argparse
import logging
import subprocess
import utils
import shlex

from constants import SRC_DIR, PANTHEON_ROOT

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Pantheon Environment Instances')

parser.add_argument('-N', '--num_env', type=int, default=4,
                    help='Number of Pantheon environment instances. '
                    'This corresponds to number of actors for RL training.')
parser.add_argument('--server_address', type=str,
                    default='unix:/tmp/rl_server_path',
                    help='RL server address - <host>:<port> or unix:<path>')
parser.add_argument('--logdir', type=str,
                    default=path.join(SRC_DIR, 'train/logs'),
                    help='Pantheon logs output directory')

src_path = path.join(PANTHEON_ROOT, 'src/experiments/test.py')


def run_pantheon(flags):
    logging.info('Starting {} Pantheon env instances'.format(flags.num_env))

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
            data_dir = path.join(flags.logdir, 'sc_%d' % job_cfg['scenario'])
            cmd_tmpl = utils.safe_format(cmd_tmpl,
                                         {'data_dir': data_dir})

            job_queue.append((job_cfg, cmd_tmpl))

    processes = []
    n = len(job_queue)
    for i in range(flags.num_env):
        job_cfg, cmd = job_queue[i % n]
        cmd_to_process = get_cmd(cmd, flags)
        logging.info('Launch cmd: {}'.format(' '.join(cmd_to_process)))
        p = subprocess.Popen(cmd_to_process, env=pantheon_env)
        processes.append(p)
    for p in processes:
        p.wait()


def get_cmd(cmd, flags):
    extra_sender_args = ' '.join([
        '--cc_env_mode=train',
        '--cc_env_rpc_address={}'.format(flags.server_address),
    ])
    cmd = shlex.split(cmd) + [
        '--extra_sender_args="{}"'.format(extra_sender_args),
    ]
    return cmd


if __name__ == "__main__":
    flags = parser.parse_args()
    run_pantheon(flags)
