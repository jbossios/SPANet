from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from spanet import JetReconstructionModel, Options
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from collections import OrderedDict
from spanet.dataset.event_info import EventInfo

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-ef", "--event_file", type=str, default="", help="Input file containing event symmetry information.")
    parser.add_argument("-tf", "--training_file", type=str, default="", help="Input file containing training data.")
    parser.add_argument("-vf", "--validation_file", type=str, default="", help="Input file containing Validation data. If not provided, will use training data split.")
    parser.add_argument("-of", "--options_file", type=str, default=None, help="JSON file with option overloads.")
    parser.add_argument("-cf", "--checkpoint", type=str, default=None, help="Optional checkpoint to load from")
    parser.add_argument("-l", "--log_dir", type=str, default=None, help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")
    parser.add_argument("-n", "--name", type=str, default="spanet_output", help="The sub-directory to create for this run.")
    parser.add_argument("-fp16", action="store_true", help="Use AMP for training.")
    parser.add_argument("-g", "--graph", action="store_true", help="Log the computation graph.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Output additional information to console and log.")
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Override batch size in hyperparameters.")
    parser.add_argument("-f", "--full_events", action='store_true', help="Limit training to only full events.")
    parser.add_argument("-p", "--limit_dataset", type=int, default=None, help="Limit dataset to only the first L percent of the data (0 - 100).")
    parser.add_argument("-r", "--random_seed", type=int, default=0, help="Set random seed for cross-validation.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs to train for")
    parser.add_argument("--gpus", type=int, default=None, help="Override GPU count in hyperparameters.")
    ops = parser.parse_args()


    # Whether or not this script version is the master run or a worker
    master = True
    if "NODE_RANK" in environ:
        master = False


    # -------------------------------------------------------------------------------------------------------
    # Create options file and load any optional extra information.
    # -------------------------------------------------------------------------------------------------------
    options = Options(ops.event_file, ops.training_file, ops.validation_file)

    if ops.options_file is not None:
        with open(ops.options_file, 'r') as json_file:
            options.update_options(json.load(json_file))


    # -------------------------------------------------------------------------------------------------------
    # Command line overrides for common option values.
    # -------------------------------------------------------------------------------------------------------
    options.verbose_output = ops.verbose
    if master and ops.verbose:
        print(f"Verbose output activated.")

    if ops.full_events:
        if master:
            print(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if ops.gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = ops.gpus

    if ops.batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = ops.batch_size

    if ops.limit_dataset is not None:
        if master:
            print(f"Overriding Dataset Limit: {limit_dataset}%")
        options.dataset_limit = ops.limit_dataset / 100

    if ops.epochs is not None:
        if master:
            print(f"Overriding Number of Epochs: {epochs}")
        options.epochs = ops.epochs

    if ops.random_seed > 0:
        options.dataset_randomization = ops.random_seed

    # -------------------------------------------------------------------------------------------------------
    # Print the full hyperparameter list
    # -------------------------------------------------------------------------------------------------------
    if master:
        options.display()

    # -------------------------------------------------------------------------------------------------------
    # Load the data
    # -------------------------------------------------------------------------------------------------------

    event_info = EventInfo.read_from_ini(ops.event_file)

    # load data if desired
    source_data = []
    with h5py.File(ops.training_file, "r") as hdf5_file:
        # get mask
        source_mask = torch.from_numpy(np.array(hdf5_file["source/mask"]))
        # get features
        for index, (feature, normalize, log_transform) in enumerate(event_info.source_features):
            temp = torch.from_numpy(np.array(hdf5_file[f'source/{feature}']))
            if log_transform:
                temp = torch.log(torch.clamp(temp, min=1e-6)) * source_mask
            if normalize:
                mean = temp[source_mask].mean()
                std = temp[source_mask].std()
                setattr(options,f"{feature}_mean", str(float(mean)))
                setattr(options,f"{feature}_mean", str(float(std)))
                temp[source_mask] = (temp[source_mask] - mean) / std
            source_data.append(temp)
        # stack data
        source_data = torch.stack(source_data,-1)
        print(f"Source data {source_data.shape}, mask {source_mask.shape}")

        # load targets
        num_events = source_data.shape[0]

        target_mask = []
        target_data = []
        for target, (jets, _) in event_info.targets.items():
            print(target, jets, _)
            target_mask.append(torch.from_numpy(np.array(hdf5_file[f"{target}/mask"])))            
            for index, jet in enumerate(jets):
                target_data.append(torch.from_numpy(np.array(hdf5_file[f"{target}/{jet}"])))
        target_mask = torch.stack(target_mask,-1)
        target_data = torch.stack(target_data,-1)
        print(target_mask.shape, target_data.shape)

        source_data_train, source_data_test, target_data_train, target_data_test, target_mask_train, target_mask_test = train_test_split(source_data, target_data, target_mask, test_size=0.25, shuffle=True)
                
        # print shapes
        print(source_data_train.shape, source_data_test.shape, target_data_train.shape, target_data_test.shape, target_mask_train.shape, target_mask_test.shape)
        
