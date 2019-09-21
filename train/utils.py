# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import yaml
import string
import itertools
from datetime import datetime

from train.constants import SRC_DIR, PANTHEON_ROOT, EXPERIMENTS_CFG


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# format 'format_string' but ignore keys that do not exist in 'key_dict'
def safe_format(format_string, key_dict):
    return string.Formatter().vformat(format_string, (), SafeDict(key_dict))


def parse_experiments():
    with open(EXPERIMENTS_CFG) as cfg:
        return yaml.load(
            safe_format(
                cfg.read(), {"src_dir": SRC_DIR, "pantheon_root": PANTHEON_ROOT}
            )
        )


expt_cfg = parse_experiments()
meta = expt_cfg["meta"]


def expand_matrix(matrix_cfg):
    input_list = []
    for variable, value_list in matrix_cfg.items():
        input_list.append([{variable: value} for value in value_list])

    ret = []
    for element in itertools.product(*input_list):
        tmp = {}
        for kv in element:
            tmp.update(kv)
        ret.append(tmp)

    return ret


def utc_date():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M")
