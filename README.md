# mvfst-rl

`mvfst-rl` is a framework for network congestion control in the QUIC transport protocol
that leverages state-of-the-art in asynchronous Reinforcement Learning training with
off-policy correction. It is built upon the following components:

1. [mvfst](https://github.com/facebookincubator/mvfst), an implementation of the IETF QUIC transport protocol.
2. [torchbeast](https://github.com/facebookresearch/torchbeast), a PyTorch implementation of asynchronous distributed deep RL.
3. [Pantheon](https://github.com/StanfordSNR/pantheon), a set of calibrated network emulators.

### Asynchronous RL Agent

![alt text](figures/rl_agent.png "RL Agent")


### Training Architecture

![alt text](figures/training_architecture.png "Training Architecture")


For more details, please refer to our [paper](https://arxiv.org/abs/1910.04054).

## Building mvfst-rl

### Ubuntu 20+

Pantheon requires Python 2 while `mvfst-rl` training requires Python 3.8+. The recommended setup is to explicitly use python2/python3 commands.

For building with training support, it is recommended to have a conda environment first:
```shell
conda create -n mvfst-rl python=3.8 -y && conda activate mvfst-rl
./setup.sh
```

If you have a previous installation and need to re-install from scratch after updating
the code, run the following commands:
```shell
conda activate base && conda env remove -n mvfst-rl
conda create -n mvfst-rl python=3.8 -y && conda activate mvfst-rl
./setup.sh --clean
```

For building `mvfst-rl` in test-only or deployment mode, run the following script.
This allows you to run a trained model exported via TorchScript purely in C++.
```
./setup.sh --inference
```

## Training

Training can be run locally as follows:
```shell
python3 -m train.train \
mode=train \
total_steps=1_000_000 \
num_actors=40 \
hydra.run.dir=/tmp/logs
```

The above starts 40 Pantheon instances in parallel that communicate with the torchbeast actors via RPC.
To see the full list of training parameters, run `python3 -m train.train --help`.

## Hyper-parameter sweeps with Hydra

`mvfst-rl` uses [Hydra](https://hydra.cc/), which in particular makes it easy to run
hyper-parameter sweeps. Here is an example showing how to run three  experiments with
different learning rates on a [Slurm](https://slurm.schedmd.com/overview.html) cluster:
```shell
python3 -m train.train \
mode=train \
test_after_train=false \
total_steps=1_000_000 \
num_actors=40 \
learning_rate=1e-5,1e-4,1e-3 \
hydra.sweep.dir='${oc.env:HOME}/tmp/logs_${now:%Y-%m-%d_%H-%M-%S}' \
hydra/launcher=_submitit_slurm -m
```

Note the following settings in the above example:
* `test_after_train=false` skips running the test mode after training. This can be useful
  for instance when the machines on the cluster have not been setup with all the libraries
  required in test mode.
* `learning_rate=1e-5,1e-4,1e-3`: this is the basic syntax to perform a parameter sweep.
* `hydra.sweep.dir='${oc.env:HOME}/tmp/logs_${now:%Y-%m-%d_%H-%M-%S}'`: the base location for all logs
  (look into the `.submitit` subfolder inside that directory to access the jobs' stdout/stderr).
* `hydra/launcher=_submitit_slurm`: the launcher used to run on Slurm. Hydra supports more
  launchers, see its [documentation](https://hydra.cc/docs/intro) for details (by default,
  the [joblib](https://hydra.cc/docs/plugins/joblib_launcher) launcher is also installed
  by `setup.sh` -- it allows running multiple jobs locally instead of on a cluster).
  Note that the launcher name must be prefixed with an underscore to match the config files
  under `config/hydra/launcher` (which you may edit to tweak launcher settings).
* `-m`: to run Hydra in [multi-run mode](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run/).

## Monitoring training behavior

The script `scripts/plotting/plot_sweep.py` can be used to plot training curves.
Refer to comments in the script's header for instructions on how to execute it.

It is also possible to use [TensorBoard](https://www.tensorflow.org/tensorboard):
the data can be found in the `train/tensorboard` subfolder of an experiment's logs directory.


## Evaluation

To test a trained model on all emulated Pantheon environments, run with `mode=test` as follows:
```
python3 -m train.train \
  mode=test \
  base_logdir=/tmp/logs
```

The above takes the `checkpoint.tar` file in `/tmp/logs`, traces the model via TorchScript,
and runs inference in C++ (without RPC).

## Pantheon logs cleanup

Pantheon generates temporary logs (in `_build/deps/pantheon/tmp`) that may take up a lot of space.
It is advised to regularly run `scripts/clean_pantheon_logs.sh` to delete them (when no experiment is running).
Note that when running jobs on a SLURM cluster, where a temporary local folder is made available to
each job in `/scratch/slurm_tmpdir/$SLURM_JOB_ID`, this folder is used instead to store the logs
(thus alleviating the need for manual cleanup).

## Contributing
We would love to have you contribute to `mvfst-rl` or use it for your research.
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
mvfst-rl is licensed under the CC-BY-NC 4.0 license, as found in the LICENSE file.

## BibTeX

```
@article{mvfstrl2019,
  title={MVFST-RL: An Asynchronous RL Framework for Congestion Control with Delayed Actions},
  author={Viswanath Sivakumar and Olivier Delalleau and Tim Rockt\"{a}schel and Alexander H. Miller and Heinrich K\"{u}ttler and Nantas Nardelli and Mike Rabbat and Joelle Pineau and Sebastian Riedel},
  year={2019},
  eprint={1910.04054},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/1910.04054},
  journal={NeurIPS Workshop on Machine Learning for Systems},
}
```
