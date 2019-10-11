The train directory contains all the tools needed to train RL-based congestion control using
IMPALA with [Pantheon](https://github.com/StanfordSNR/pantheon) as the network emulator.

Each Pantheon environment instance can be configured to run with a different emulated network setting obtained from
https://github.com/StanfordSNR/observatory/blob/master/src/scripts/experiments.yml. The relevant
trace files in traces/ are copied from https://github.com/StanfordSNR/observatory/tree/master/traces.
