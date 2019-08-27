import os
from os import path

SRC_DIR = path.abspath(path.join(path.dirname(__file__), os.pardir))
PANTHEON_ROOT = path.join(SRC_DIR, "_build/deps/pantheon")
EXPERIMENTS_CFG = path.join(SRC_DIR, "train/experiments.yml")
TORCHBEAST_ROOT = path.join(SRC_DIR, "third-party/torchbeast")
