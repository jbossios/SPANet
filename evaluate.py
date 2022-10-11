#! /usr/bin/env python

'''
Evaluation scipt for Spanet architecture. 
'''

# python packages
import argparse
import os
from glob import glob
import h5py
from numpy import ndarray as Array
import numpy as np

# multiprocessing
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
    #mp.set_start_method('forkserver', force=True)
except RuntimeError:
    print("Could not set multiprocessing to spawn")
    exit()
    #pass

# SOME HACK ON mac
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# spanet
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.evaluation import predict_on_test_dataset, load_model

def main():

    # user options
    ops = options() 

    # check if multiprocessing should be done
    input_files = handleInput(ops.inFile)

    # make output dir
    if not os.path.isdir(ops.outDir):
        os.makedirs(ops.outDir)

    # create evaluation job dictionaries
    config  = []
    for inFileName in input_files:
        
        # make output file name
        outFileName = os.path.join(ops.outDir, os.path.basename(inFileName)).replace(".h5",f"_spanet.h5")
        if os.path.isfile(outFileName) and not ops.doOverwrite:
            print(f"File already exists not evaluating on: {outFileName}")
            continue

        # create outfile tag
        config.append({
            "log_dir" : ops.log_dir,
            "test_file" : inFileName,
            "EVENT_FILE" : ops.event_file,
            "batch_size" : None,
            "gpu" : False,
            "output_file" : outFileName
        })

    # launch jobs
    if ops.ncpu == 1:
        for conf in config:
            evaluate(conf)
    else:
        results = mp.Pool(ops.ncpu).map(evaluate, config)

def options():
    ''' argument parser to handle inputs from the user '''
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-l", "--log_dir", help="Pretrained weights to evaluate with.", default=None, required=True)
    parser.add_argument("-i", "--inFile", help="Data file to evaluate on.", default=None, required=True)
    parser.add_argument("-e", "--event_file", help="Event file configuration.", default=None, required=True)
    parser.add_argument("-j", "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("-o", "--outDir", help="Directory to save evaluation output to.", default="./")
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files.")
    return parser.parse_args()

def handleInput(data):
    # otherwise return 
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted(os.listdir(data))
    elif "*" in data:
        return sorted(glob(data))
    return []

def create_hdf5_output(input_file: str,
                       output_file: str,
                       dataset: JetReconstructionDataset,
                       full_predictions: Array,
                       full_classifications: Array):

    # compute masses
    with h5py.File(input_file, "r") as hf:
        x = np.stack([np.array(hf[f'source/{key}']) for key in ["pt","eta","phi","mass"]],-1)
    px = x[:,:,0] * np.cos(x[:,:,2])
    py = x[:,:,0] * np.sin(x[:,:,2])
    pz = x[:,:,0] * np.sinh(x[:,:,1])
    e = np.sqrt(x[:,:,3]**2 + px**2 + py**2 + pz**2) # m^2 + p^2
    x = np.stack([e,px,py,pz],-1)

    # pick up predictions
    g1 = np.stack([np.array(full_predictions[0][:,k]) for k in range(full_predictions[0].shape[1])],-1)
    g2 = np.stack([np.array(full_predictions[1][:,k]) for k in range(full_predictions[0].shape[1])],-1)
    # pickup correct jets and sum
    g1p = np.take_along_axis(x, np.expand_dims(g1,-1).repeat(4,-1), 1)
    g2p = np.take_along_axis(x, np.expand_dims(g2,-1).repeat(4,-1), 1)
    # create gluino masses
    g1m = g1p.sum(1)
    g2m = g2p.sum(1)
    g1m = np.sqrt(g1m[:,0]**2 - g1m[:,1]**2 - g1m[:,2]**2 - g1m[:,3]**2)
    g2m = np.sqrt(g2m[:,0]**2 - g2m[:,1]**2 - g2m[:,2]**2 - g2m[:,3]**2)
    gluino_mass_pred = np.stack([g1m, g2m], -1)
    # create neutralino masses
    neutralino_mass_pred = None
    if full_predictions[0].shape[1] == 5:
        neutralino_mass_pred = np.stack([g1p[:,[2,3,4]],g2p[:,[2,3,4]]],1).sum(2)
        neutralino_mass_pred = np.sqrt(neutralino_mass_pred[:,:,0]**2 - neutralino_mass_pred[:,:,1]**2 - neutralino_mass_pred[:,:,2]**2 - neutralino_mass_pred[:,:,3]**2)

    print(f"Creating output file at: {output_file}")
    with h5py.File(output_file, 'w') as output:
        output.create_dataset(f"source/mask", data=dataset.source_mask)
        for i, (feature_name, _, _) in enumerate(dataset.event_info.source_features):
            output.create_dataset(f"source/{feature_name}", data=dataset.source_data[:, :, i])

        for i, (particle_name, (jets, _)) in enumerate(dataset.event_info.targets.items()):
            output.create_dataset(f"{particle_name}/mask", data=full_classifications[i])
            for k, jet_name in enumerate(jets):
                output.create_dataset(f"{particle_name}/{jet_name}", data=full_predictions[i][:, k])
        
        # save mass predictions
        output.create_dataset("gluino_mass_pred", data=gluino_mass_pred)
        if neutralino_mass_pred is not None:
            output.create_dataset("neutralino_mass_pred", data=neutralino_mass_pred)

def evaluate(c):
    ''' perform the full model evaluation '''

    print(f"Evaluating on {c['test_file']}")
    model = load_model(c["log_dir"], c["test_file"], c["EVENT_FILE"], c["batch_size"], c["gpu"], num_workers = 0)
    full_predictions, full_classifications, *_ = predict_on_test_dataset(model, c["gpu"])
    create_hdf5_output(c["test_file"], c["output_file"], model.testing_dataset, full_predictions, full_classifications)  


if __name__ == "__main__":
    main()
