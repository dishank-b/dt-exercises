#!/usr/bin/env python3
import os
import sys

SRC_PATH = "../exercise_ws/src/object_detection/include/object_detection/faster_rcnn/"
sys.path.append(SRC_PATH)

import numpy as np
import torch
import random 

from src.engine.trainer import General_Solver
from src.config import Cfg as cfg  # Configuration file

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib
import argparse
import time

import torchvision

matplotlib.use('agg')

#----- Initial paths setup and loading config values ------ #

print("\n--- Setting up Trainer/Tester \n")

ap = argparse.ArgumentParser()
ap.add_argument("-name",
                "--name",
                required=True,
                help="Comments for the experiment")
ap.add_argument("-config",
                "--config",
                required=False,
                default=None,
                help="Give your experiment configuration file")
ap.add_argument("-mode", "--mode", required=True, choices=['train', 'test'])
ap.add_argument("-weights", "--weights", default=None)
ap.add_argument("-resume", "--resume", default=False)
ap.add_argument("-epoch", "--epoch")
args = ap.parse_args()

if args.config:
    print("Loading exp config")
    cfg.merge_from_file(os.path.join(SRC_PATH, args.config))
cfg.freeze()


mode = args.mode

# Config Operations

#---------Training/Testing Cycle-----------#
epochs = cfg.TRAIN.EPOCHS
saving_freq = cfg.TRAIN.SAVE_MODEL_EPOCHS
solver = General_Solver(cfg, mode, args)
if mode == "train":
    solver.train(epochs, saving_freq)
else:
    solver.test()
