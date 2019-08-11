import yaml
import string
import itertools
from datetime import datetime

from constants import SRC_DIR, PANTHEON_ROOT, EXPERIMENTS_CFG


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