import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
#matplotlib inline

from utils.plot_utils import create_overlap
from developing_suite import *


str_args = ["--model=UResNet", #UResNet", 
            "--save-dir=./runs/",
            "--save-model=last",
            "--mode=train",
            "--workers=1",
            "--data=lenticular_full_image", #lenticular_square_patch",
            "--datafolder=./dataset/example_dataset",
            "--epochs=1000",
            "--logstep-train=10",
            "--batch-size=1",
            "--train-val-ratio=0.9",
            "--optimizer=adam",
            "--lr=0.001",
            "--lr-scheduler=step",
            "--lr-step=10",
            "--lr-gamma=0.5",
            "--loss=CE"]

args = parser.parse_args(str_args)
developingSuite = DevelopingSuite(args)

developingSuite.train_and_eval()
