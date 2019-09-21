# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse


def add_args(parser):
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "test_local", "trace"],
        help="test -> remote test, test_local -> local inference.",
    )
    parser.add_argument(
        "--traced_model",
        default="traced_model.pt",
        help="File to write torchscript traced model to (for training) "
        "or read from (for local testing).",
    )
