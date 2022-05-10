from collections import defaultdict
from typing import Optional, List, Any, Dict
from os import getcwd, makedirs, environ
from sys import stderr, stdout
import json
import numpy as np
import torch
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Copy Secret
os.system('cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_1000')
os.system('chmod 600 /tmp/krb5cc_1000')
os.system('cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0')
os.system('chmod 600 /tmp/krb5cc_0')
os.system('ls /tmp')
os.system('ls /eos/user/j/jbossios/SUSY/SpaNet/PackageForKubeflow/SPANet/')

import sys
sys.path.insert(1, '/eos/user/j/jbossios/SUSY/SpaNet/PackageForKubeflow/SPANet/')
from spanet import JetReconstructionModel, Options
from spanet.dataset.evaluator import SymmetricEvaluator, EventInfo
from spanet.evaluation import predict_on_test_dataset, load_model

# Read arguments
parser = argparse.ArgumentParser(description='Spanet Params')
parser.add_argument('--epochs',                       type=int,   default=50,      help='Number of epochs')
parser.add_argument('--learning_rate',                type=float, default=0.0001,  help='Learning rate')
parser.add_argument('--batch_size',                   type=int,   default=2048,    help='Batch size')
parser.add_argument('--dropout',                      type=float, default=0.0,     help='Dropout percentage')
parser.add_argument('--l2_penalty',                   type=float, default=0.0002,  help='L_2 weight normalization')
parser.add_argument('--hidden_dim',                   type=int,   default=128,     help='Dimensions used internally by all hidden layers / transformers')
parser.add_argument('--initial_embedding_dim',        type=int,   default=16,      help='Hidden dimensionality of the first embedding layer')
parser.add_argument('--num_branch_encoder_layers',    type=int,   default=3,       help='Branch encoder count  (try with 1,2,3,4,5,6)')
parser.add_argument('--num_encoder_layers',           type=int,   default=6,       help='Central encoder count (try with 4,8,12)')
parser.add_argument('--partial_events',               type=int,   default=0,       help='Partial Event training')
parser.add_argument('--num_attention_heads',          type=int,   default=8,       help='try with 4 and 12 too')
parser.add_argument('--num_branch_embedding_layers',  type=int,   default=5,       help='Try with 4 and 6')
parser.add_argument('--num_jet_encoder_layers',       type=int,   default=1,       help='Try with 0 and 1')
parser.add_argument('--event_file',                   type=str,   default='/eos/user/j/jbossios/SUSY/SpaNet/PackageForKubeflow/SPANet/event_files/signal.ini')
parser.add_argument('--training_file',                type=str,   default='/eos/user/j/jbossios/SUSY/SpaNet/SpaNetInputs/signal_training_v4.h5')
parser.add_argument('--validation_file',              type=str,   default='') 
parser.add_argument('--testing_file',                 type=str,   default='/eos/user/j/jbossios/SUSY/SpaNet/SpaNetInputs/signal_testing_v4.h5') 

args = parser.parse_args()

options = Options(args.event_file, args.training_file, args.validation_file)
# Override default options
options.learning_rate               = args.learning_rate
options.batch_size                  = args.batch_size
options.dropout                     = args.dropout
options.l2_penalty                  = args.l2_penalty
options.hidden_dim                  = args.hidden_dim
options.initial_embedding_dim       = args.initial_embedding_dim
options.num_branch_encoder_layers   = args.num_branch_encoder_layers
options.num_encoder_layers          = args.num_encoder_layers
options.partial_events              = args.partial_events
options.combine_pair_loss           = 'softmin'
options.epochs                      = args.epochs
options.num_attention_heads         = args.num_attention_heads
options.num_branch_embedding_layers = args.num_branch_embedding_layers
options.num_gpu                     = 1 # FIXME
options.optimizer                   = 'AdamW'
options.testing_file                = args.testing_file
# print options
options.display()

# Create the initial model on the CPU
model = JetReconstructionModel(options)

# If we are using more than one gpu, then switch to DDP training
distributed_backend = 'dp' if options.num_gpu > 1 else None

# Write metrics to EOS for postprocessing
model_output_dir = '/eos/user/j/jbossios/SUSY/SpaNet/Katib/Outputs/'

# get random number
import random
randN = random.random()

# Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
#log_dir = getcwd()
log_dir = model_output_dir + 'Model_{}'.format(randN)
if not os.path.exists(log_dir+'/spanet_output'):
    os.makedirs(log_dir+'/spanet_output')
logger  = TensorBoardLogger(save_dir=log_dir, name='spanet_output', log_graph=False)

# Create the checkpoint for this training run. We will save the best validation networks based on 'accuracy'
checkpoint_callback = ModelCheckpoint(verbose=options.verbose_output,
                                      monitor='validation_accuracy',
                                      save_top_k=1,
                                      mode='max',
                                      save_last=True)

learning_rate_callback = LearningRateMonitor()

# Create the final pytorch-lightning manager
trainer = pl.Trainer(logger=logger,
                     max_epochs=options.epochs,
                     callbacks=[checkpoint_callback, learning_rate_callback],
                     resume_from_checkpoint=None,
                     distributed_backend=distributed_backend,
                     gpus=options.num_gpu if options.num_gpu > 0 else None,
                     track_grad_norm=2 if options.verbose_output else -1,
                     gradient_clip_val=options.gradient_clip,
                     weights_summary='full' if options.verbose_output else 'top',
                     precision=32)

# Save the current hyperparameters to a json file in the checkpoint directory
print(f"Training Version {trainer.logger.version}")
makedirs(trainer.logger.log_dir, exist_ok=True)
with open(trainer.logger.log_dir + "/options.json", 'w') as json_file:
    json.dump(options.__dict__, json_file, indent=4)

trainer.fit(model)

def evaluate_model(model: JetReconstructionModel, cuda: bool = False):
    predictions, _, targets, masks, num_jets = predict_on_test_dataset(model, cuda)

    event_info = EventInfo.read_from_ini(model.options.event_info_file)
    evaluator = SymmetricEvaluator(event_info)

    minimum_jet_count = num_jets.min()
    jet_limits = [f"== {minimum_jet_count}",
                  f"== {minimum_jet_count + 1}",
                  f">= {minimum_jet_count + 2}",
                  None]

    results = {}
    for jet_limit_name in jet_limits:
        limited_predictions = predictions
        limited_targets = targets
        limited_masks = masks

        if jet_limit_name is not None:
            jet_limit = eval("num_jets {}".format(jet_limit_name))
            limited_predictions = [p[jet_limit] for p in limited_predictions]
            limited_targets = [t[jet_limit] for t in limited_targets]
            limited_masks = [m[jet_limit] for m in limited_masks]

        results[jet_limit_name] = evaluator.full_report_string(limited_predictions, limited_targets, limited_masks)
        results[jet_limit_name]["event_jet_proportion"] = 1.0 if jet_limit_name is None else jet_limit.mean()

    return results, jet_limits

# Evaluate

# load model
log_directory = log_dir + '/spanet_output/version_0'
model         = load_model(log_directory, options.testing_file, options.event_info_file, options.batch_size, True if options.num_gpu > 0 else False)

#if options.num_gpu > 0:
#    model = model.cuda()
results, jet_limits = evaluate_model(model, True if options.num_gpu > 0 else False)
reco_efficiency    = results[None]["2g/event_purity"]

with open(model_output_dir + 'metrics_custom_{}.txt'.format(randN), 'w') as f:
    f.write('learning_rate               = {}\n'.format(options.learning_rate))
    f.write('hidden_dim                  = {}\n'.format(options.hidden_dim))
    f.write('initial_embedding_dim       = {}\n'.format(options.initial_embedding_dim))
    f.write('num_branch_encoder_layers   = {}\n'.format(options.num_branch_encoder_layers))
    f.write('num_encoder_layers          = {}\n'.format(options.num_encoder_layers))
    f.write('num_attention_heads         = {}\n'.format(options.num_attention_heads))
    f.write('num_branch_embedding_layers = {}\n'.format(options.num_branch_embedding_layers))
    f.write('reco_efficiency             = {}\n'.format(reco_efficiency))

# Write metric for Katib
model_output_dir = '/model_outputs'

if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

with open(model_output_dir + '/metrics_custom.txt', 'w') as f:
    f.write('reco_efficiency=' + str(reco_efficiency))
