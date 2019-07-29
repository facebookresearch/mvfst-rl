import os
from os import path

SRC_DIR = path.abspath(path.join(path.dirname(__file__), os.pardir))
PANTHEON_ROOT = path.abspath(path.join(SRC_DIR, '_build/deps/pantheon'))
EXPERIMENTS_CFG = path.abspath(path.join(SRC_DIR, 'train/experiments.yml'))
