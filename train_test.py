from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json

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
    options = Options(event_file, training_file, validation_file)

    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))


    # -------------------------------------------------------------------------------------------------------
    # Command line overrides for common option values.
    # -------------------------------------------------------------------------------------------------------
    options.verbose_output = verbose
    if master and verbose:
        print(f"Verbose output activated.")

    if full_events:
        if master:
            print(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if limit_dataset is not None:
        if master:
            print(f"Overriding Dataset Limit: {limit_dataset}%")
        options.dataset_limit = limit_dataset / 100

    if epochs is not None:
        if master:
            print(f"Overriding Number of Epochs: {epochs}")
        options.epochs = epochs

    if random_seed > 0:
        options.dataset_randomization = random_seed

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
    with h5py.File(training_file, "r") as hdf5_file:
        # get mask
        source_mask = torch.from_numpy(hdf5_file["source/mask"])
        # get features
        for index, (feature, normalize, log_transform) in enumerate(event_info.source_features):
            temp = hdf5_file[f'source/{feature}']
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
        targets = OrderedDict()
        for target, (jets, _) in event_info.targets.items():
            target_mask = torch.from_numpy(hdf5_file[f"{target}/mask"])
            target_data = torch.empty(len(jets), self.num_events, dtype=torch.int64)

            for index, jet in enumerate(jets):
                hdf5_file[f"{target}/{jet}"].read_direct(target_data[index].numpy())

            target_data = target_data.transpose(0, 1)
            target_data = target_data[limit_index]

            targets[target] = (target_data, target_mask)

        print(targets.keys())

