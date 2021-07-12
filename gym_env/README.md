# MvFstEnv

Gym-Compatible environment for congestion control.

## Contents

1. [Introduction](#Introduction)

2. [Setup](#Setup)

3. [Usage](#Usage)

4. [Local Development](#Local-Development)

## Introduction

MvFstEnv provides a gym-compatible env for congestion control, with support for multi-task settings thanks to the [MTEnv](https://github.com/facebookresearch/mtenv) interface.
This README provides documentation specifically for MvFstEnv. For details about the mvfst-rl project it is based on, refer to the README [here](https://github.com/facebookresearch/mvfst-rl/tree/mtenv).

## Setup

* Clone the repository: `git clone git@github.com:facebookresearch/mvfst-rl.git`.

* Checkout the `mtenv` branch.

* Setup mvfst-rl using the instructions from [README](https://github.com/facebookresearch/mvfst-rl/tree/mtenv#building-mvfst-rl)

* Install dependencies: `pip install -r gym_env/requirements.txt`

## Usage

* We provide an example script to use the environment in `example.py`.

* Invoke `example.py` with similar arguments that you would use for running mvfst-rl. 

* Since we do not run any learner, the arguments related to the learners are ignored.

* We introduce two new config variables: 
    
    * `env_configs` (similar to `train_jobs` or `eval_jobs`)

    * `env_ids` (similar to `train_job_ids` or `eval_job_ids`). These are used to select configs in the `env_configs`.

* `example.py` shows an example of using both the single task setup and the multi-task setup. The choice between single vs multitask is made on the basis of size of `env_ids`. If the size is 1, a standard gym-compatible env is created. If size > 1, a [MTEnv](https://github.com/facebookresearch/mtenv) environment is created.

* Example Command for running with a single env:

```
python3 -m gym_env.example \
mode=train \
jobs@env_configs=fixed_0_5 \
num_actors=1 \
env_ids="[0]" \
cc_env_norm_bytes=1000000 \
cc_env_norm_ms=10000 \
cc_env_history_size=16 \
cc_env_reward_delay_factor=0.75 \
cc_env_reward_packet_loss_factor=0 \
cc_env_reward_formula=log_ratio \
cc_env_reward_delay_offset=0 \
cc_env_reward_min_throughput_ratio=0.9 \
cc_env_reward_n_packets_offset=1
```

* Example Command for running with a MTEnv:

```
python3 -m gym_env.example \
mode=train \
jobs@env_configs=fixed_0_5 \
num_actors=1 \
env_ids="[0,1,2,3]" \
cc_env_norm_bytes=1000000 \
cc_env_norm_ms=10000 \
cc_env_history_size=16 \
cc_env_reward_delay_factor=0.75 \
cc_env_reward_packet_loss_factor=0 \
cc_env_reward_formula=log_ratio \
cc_env_reward_delay_offset=0 \
cc_env_reward_min_throughput_ratio=0.9 \
cc_env_reward_n_packets_offset=1
```

## Local Development

* To run mypy, use the config file `gym_env/setup.cfg` i.e. `mypy --config-file gym_env/setup.cfg  gym_env`
