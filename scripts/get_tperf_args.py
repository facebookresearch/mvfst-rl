#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility script to obtain the arguments to be provided to tperf to test a trained model.

Usage:
    get_tperf_args.py <exp_folder>
where `exp_folder` is the base experiment folder.
"""

import argparse
import json
import logging
import os
import subprocess
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

TPERF_SETTINGS = [
    "cc_env_mode",
    "cc_env_model_file",
    "cc_env_job_count",
]

# These arguments are to be read from the experiment training configuration, to be
# forwarded to the `tperf` executable.
TPERF_FLAGS = [
    "cc_env_agg",
    "cc_env_time_window_ms",
    "cc_env_fixed_window_size",
    "cc_env_use_state_summary",
    "cc_env_history_size",
    "cc_env_norm_ms",
    "cc_env_norm_bytes",
    "cc_env_actions",
    "cc_env_reward_log_ratio",
    "cc_env_reward_throughput_factor",
    "cc_env_reward_throughput_log_offset",
    "cc_env_reward_delay_factor",
    "cc_env_reward_delay_log_offset",
    "cc_env_reward_packet_loss_factor",
    "cc_env_reward_packet_loss_log_offset",
    "cc_env_reward_max_delay",
    "cc_env_fixed_cwnd",
    "cc_env_min_rtt_window_length_us",
]


def get_cc_env_actions(actions: List[str]) -> str:
    """
    Obtain list of actions in the format suitable to tperf.

    In 'meta.json', actions are given as a list of strings, but somehow the "+"
    sign in front of a positive integer is lost. We restore it here.
    """
    tokens = []
    for action in actions:
        if action == "0" or any(action.startswith(c) for c in ["+", "-", "*", "/"]):
            tokens.append(action)
        else:
            # Verify that this is indeed a positive integer.
            try:
                int(action)
            except ValueError:
                raise NotImplementedError(f"Unsupported action: {action}")
            tokens.append(f"+{action}")
    return ",".join(tokens)


def init_logger() -> None:
    """Initialize logger"""
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger.setLevel(logging.INFO)


def parse_args() -> Any:
    """Parse commnad line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    return parser.parse_args()


def to_cmd_line(
    flags: Dict[str, Any], exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Convert experiment flags into command-line arguments.
    """
    exclude = set() if exclude is None else set(exclude)
    args: List[str] = [f"{k}={to_str(v)}" for k, v in flags.items() if k not in exclude]
    return args


def to_str(val: Any) -> str:
    """
    Convert a value to its string representation to be used on the command line.
    """
    if val is None:
        return "null"
    elif isinstance(val, str):
        return val
    elif isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, list):
        return f"[{','.join(to_str(v) for v in val)}]"
    else:
        raise NotImplementedError(
            f"Unsupported type '{type(val).__name__}' with value: `{val}`"
        )


def trace(path: Path, flags: Dict[str, Any]) -> None:
    """
    Trace the trained model.
    """
    cwd = Path.cwd()
    script_dir = Path(__file__).absolute().parent
    os.chdir(script_dir / "..")
    try:
        cmd_line = to_cmd_line(flags, exclude=["mode"])
        cmd = ["python", "-m", "train.train", "mode=trace"] + cmd_line
        logger.info("Tracing model, command:\n  " + " ".join(cmd))
        subprocess.check_call(cmd)
    finally:
        os.chdir(cwd)


def main() -> int:
    """
    Script entry point.
    """
    init_logger()
    args = parse_args()

    # Load settings.
    path = Path(args.path)
    meta_path = path / "train" / "meta.json"
    assert meta_path.is_file(), "meta.json not found!"
    with meta_path.open() as f:
        meta = json.load(f)
    flags = meta["flags"]

    # Check that the config is valid.
    assert not flags[
        "use_job_id_in_actor"
    ], "providing the job ID is not currently supported"

    # Trace model if needed.
    traced_path = path / "traced_model.pt"
    if traced_path.exists():
        logger.info("Traced model already exists")
    else:
        trace(path, flags)

    # Extract relevant tperf settings.
    tperf_args = {f: flags[f] for f in TPERF_FLAGS}

    # Post-process / add arguments.
    tperf_args.update(
        {
            "congestion": "rl",
            "cc_env_job_count": -1,
            "cc_env_mode": "local",
            "cc_env_model_file": str(traced_path),
            "cc_env_actions": get_cc_env_actions(flags["cc_env_actions"]),
        }
    )

    # Output tperf args.
    tperf_cmd = " ".join(f"-{k}='{to_str(v)}'" for k, v in tperf_args.items())
    logger.info("tperf command line arguments:\n%s", tperf_cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
