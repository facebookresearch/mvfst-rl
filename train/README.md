## About

--------

The train directory contains all the tools needed to test RL algorithms 
using [pantheon](https://github.com/fairinternal/pantheon). 

You can configure the each run of test.py instance by emulating some type of the network. 
This can be done using trace files. The example configurations for 18 types of network with
their trace files are done using [Stanford observatory repository](https://github.com/StanfordSNR/observatory).

## Run experiments

------------------

To run experiments just write the following command

`./train/remoteenv.py --start_port=60000 --num_servers=18`

The arguments can be different. If num_servers is more than number of network to emulate, then 
it goes round-robin over the emulation examples. 

