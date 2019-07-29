## About

The train directory contains all the tools needed to train RL-based congestion control using
IMPLALA with [Pantheon](https://github.com/fairinternal/pantheon) as the network emulator.

Each Pantheon environment instance (pantheon/src/experiments/test.py) can be configured to run
with a different emulated network setting as obtained from
https://github.com/StanfordSNR/observatory/blob/master/src/scripts/experiments.yml. The relevant
trace files in traces/ are copied from https://github.com/StanfordSNR/observatory/tree/master/traces.

## Run Experiments

To start the Panthoen env for training, use the following command. This starts `N`
instances of the environment, each one correponding to an actor in the IMPALA framework.
Underneath, each env instantiates a rpcenv server to communicate with the learner.

`./train/pantheon_env.py --start_port=60000 [-N=18]`

Each instance corrsponds to a separate emulated network as taken from experiments.yml.
If `N` is greater than the number of available emulated network types, we round-robin.
