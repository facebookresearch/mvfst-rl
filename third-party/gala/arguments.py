# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse

import torch


def get_args(arg_dict=None):
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--sync-freq',
        type=int,
        default=0,
        help='max amount of message staleness for local gossip')
    parser.add_argument(
        '--num-learners',
        type=int,
        default=1,
        help='number of learners to stack on device')
    parser.add_argument(
        '--num-peers',
        type=int,
        default=1,
        help='number of peers to communicate with in each iteration')
    parser.add_argument(
        '--lr',
        type=float,
        default=7e-4,
        help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-procs-per-learner',
        type=int,
        default=16,
        help='num simulators per learner (default: 16)')
    parser.add_argument(
        '--max-steps',
        type=int,
        default=int(10e3),
        help='max episode length (default: 10,000)')
    parser.add_argument(
        '--num-steps-per-update',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, measured in environment steps (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, measured in environment steps (default: 100)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of total environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--eval-log-dir',
        default='/tmp/gym/eval/',
        help='directory to save agent eval-logs (default: /tmp/gym/eval/)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--cuda-device',
        type=int,
        default=0,
        help='index of cuda device to use')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    args = parser.parse_args(arg_dict)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
