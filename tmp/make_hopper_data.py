###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
from lib.plotting import *

#from lib.rnn_baselines import *
#from lib.ode_rnn import *
#from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
#from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
#from lib.diffeq_solver import DiffeqSolver
from mujoco_physics import HopperPhysics

#from lib.utils import compute_loss_all_batches

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n', type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz',
                    action='store_true',
                    help="Show plots while training")
parser.add_argument('--save',
                    type=str,
                    default='experiments/',
                    help="Path for save checkpoints")
parser.add_argument(
    '--load',
    type=str,
    default=None,
    help=
    "ID of the experiment to load for evaluation. If None, run a new experiment."
)
parser.add_argument('-r',
                    '--random-seed',
                    type=int,
                    default=1991,
                    help="Random_seed")

parser.add_argument(
    '--dataset',
    type=str,
    default='hopper',
    help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument(
    '-s',
    '--sample-tp',
    type=float,
    default=None,
    help="Number of time points to sub-sample."
    "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample"
)

parser.add_argument(
    '-c',
    '--cut-tp',
    type=int,
    default=None,
    help=
    "Cut out the section of the timeline of the specified length (in number of points)."
    "Used for periodic function demo.")

parser.add_argument(
    '--quantization',
    type=float,
    default=0.1,
    help="Quantization on the physionet dataset."
    "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min"
)
parser.add_argument(
    '--extrap',
    action='store_true',
    help=
    "Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t',
                    '--timepoints',
                    type=int,
                    default=100,
                    help="Total number of time-points")
parser.add_argument(
    '--max-t',
    type=float,
    default=5.,
    help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight',
                    type=float,
                    default=0.01,
                    help="Noise amplitude for generated traejctories")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # experimentID = args.load
    # if experimentID is None:
    #     # Make a new experiment ID
    #     experimentID = int(SystemRandom().random() * 100000)
    # ckpt_path = os.path.join(args.save,
    #                          "experiment_" + str(experimentID) + '.ckpt')

    # start = time.time()
    # print("Sampling dataset of {} training examples".format(args.n))

    # input_command = sys.argv
    # ind = [
    #     i for i in range(len(input_command)) if input_command[i] == "--load"
    # ]
    # if len(ind) == 1:
    #     ind = ind[0]
    #     input_command = input_command[:ind] + input_command[(ind + 2):]
    # input_command = " ".join(input_command)

    utils.makedirs("results/")

    ##################################################################
    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]

    data_obj["dataset_obj"]
    data_obj["train_dataloader"]
    batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
    batch_dict.keys()
    batch_dict['observed_data'].shape
    # batch_dict['observed_data'] batch_size, time, coordinates
    data_obj["dataset_obj"].visualize(batch_dict['observed_data'][2])

    batch_dict['observed_tp']  #.shape
    batch_dict['data_to_predict'].shape
    batch_dict['tp_to_predict']  #.shape

    data_obj["test_dataloader"]
    data_obj["input_dim"]
    data_obj["n_train_batches"]
    data_obj["n_test_batches"]

    print("Data downloaded")

    # Save data
    # Need to add individual randomness in trajectory of data
