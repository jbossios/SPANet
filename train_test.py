
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# python imports
from argparse import ArgumentParser
# from typing import Optional
import os
# from os import getcwd, makedirs, environ
import json
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import gc

# pytorch imports
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping, ModelCheckpoint, LearningRateMonitor

# spanet imports
from spanet import Options
from spanet.dataset.event_info import EventInfo
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork

# global variables
import logging

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

    # logger
    logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
    log = logging.getLogger('evaluate')

    # Whether or not this script version is the master run or a worker
    master = True
    if "NODE_RANK" in os.environ:
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
        log.info(f"Verbose output activated.")

    if ops.full_events:
        if master:
            log.info(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if ops.gpus is not None:
        if master:
            log.info(f"Overriding GPU count: {ops.gpus}")
        options.num_gpu = ops.gpus

    if ops.batch_size is not None:
        if master:
            log.info(f"Overriding Batch Size: {ops.batch_size}")
        options.batch_size = ops.batch_size

    if ops.limit_dataset is not None:
        if master:
            log.info(f"Overriding Dataset Limit: {ops.limit_dataset}%")
        options.dataset_limit = ops.limit_dataset / 100

    if ops.epochs is not None:
        if master:
            log.info(f"Overriding Number of Epochs: {ops.epochs}")
        options.epochs = ops.epochs

    if ops.random_seed > 0:
        options.dataset_randomization = ops.random_seed

    # -------------------------------------------------------------------------------------------------------
    # Print the full hyperparameter list
    # -------------------------------------------------------------------------------------------------------
    # if master:
        # options.display()

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
        log.info(f"Source data {source_data.shape}, mask {source_mask.shape}")

        # load targets
        num_events = source_data.shape[0]

        target_mask = []
        target_data = []
        for target, (jets, _) in event_info.targets.items():
            log.debug(f"{target}, {jets}, {_}")
            target_mask.append(torch.from_numpy(np.array(hdf5_file[f"{target}/mask"])))            
            for index, jet in enumerate(jets):
                target_data.append(torch.from_numpy(np.array(hdf5_file[f"{target}/{jet}"])))
        target_mask = torch.stack(target_mask,-1)
        target_data = torch.stack(target_data,-1)
        log.info(f"Target data {target_data.shape}, mask {target_mask.shape}")

        N = source_data.shape[0]
        source_data_train, source_data_test, source_mask_train, source_mask_test, target_data_train, target_data_test, target_mask_train, target_mask_test = train_test_split(source_data[:N], source_mask[:N], target_data[:N], target_mask[:N], test_size=0.25, shuffle=True)
                
        # print shapes
        log.debug(f"(Train, Test): source data ({source_data_train.shape}, {source_data_test.shape}), source mask ({source_mask_train.shape}, {source_mask_test.shape}), target data ({target_data_train.shape}, {target_data_test.shape}), target mask ({target_mask_train.shape}, {target_mask_test.shape})")
    
    dataloader_settings = {
        "batch_size": options.batch_size,
        "pin_memory": options.num_gpu > 0,
        "num_workers": options.num_dataloader_workers,
        "prefetch_factor": 2,
        "shuffle" : False
    }
    train_dataloader = DataLoader(TensorDataset(source_data_train, source_mask_train, target_data_train, target_mask_train), **dataloader_settings)
    val_dataloader   = DataLoader(TensorDataset(source_data_test,  source_mask_test,  target_data_test,  target_mask_test), **dataloader_settings)

    # set options
    setattr(options, "steps_per_epoch", source_data_train.shape[0] // options.batch_size)
    setattr(options, "total_steps", options.steps_per_epoch * options.epochs)
    setattr(options, "warmup_steps", int(round(options.steps_per_epoch * options.learning_rate_warmup_epochs)))

    # cleanup
    del source_data_train, source_data_test, source_mask_train, source_mask_test, target_data_train, target_data_test, target_mask_train, target_mask_test, source_data, source_mask, target_data, target_mask
    gc.collect()

    # -------------------------------------------------------------------------------------------------------
    # Begin the training loop
    # -------------------------------------------------------------------------------------------------------

    # Create the initial model on the CPU
    model = JetReconstructionNetwork(options)

    # If we are using more than one gpu, then switch to DDP training
    distributed_backend = 'dp' if options.num_gpu > 1 else None
    # distributed_backend = 'ddp' if options.num_gpu > 1 else None

    # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
    log_dir = os.getcwd() if ops.log_dir is None else ops.log_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=ops.name, log_graph=ops.graph)

    # callbacks
    callbacks = [
        # ModelSummary(max_depth=-1),
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0, patience=10, verbose=True),
        ModelCheckpoint(verbose=options.verbose_output, monitor="validation_accuracy", save_top_k=1, mode="max", save_last=True),
        LearningRateMonitor()
    ]

    # Create the final pytorch-lightning manager
    trainer = pl.Trainer(logger=logger,
                         max_epochs=ops.epochs,
                         callbacks=callbacks,
                         resume_from_checkpoint=ops.checkpoint,
                         # distributed_backend=distributed_backend,
                         gpus=options.num_gpu if options.num_gpu > 0 else None,
                         track_grad_norm=2 if options.verbose_output else -1,
                         gradient_clip_val=options.gradient_clip,
                         weights_summary='full' if options.verbose_output else 'top',
                         precision=16 if ops.fp16 else 32)

    # Save the current hyperparameters to a json file in the checkpoint directory
    if master:
        log.info(f"Training Version {trainer.logger.version}")
        os.makedirs(trainer.logger.log_dir, exist_ok=True)
        with open(trainer.logger.log_dir + "/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)

    trainer.fit(model, train_dataloader, val_dataloader)

    # save model
    trainer.save_checkpoint(os.path.join(logger.root_dir,f"version_{logger.version}","finalWeights.ckpt"))
