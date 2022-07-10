#! /usr/bin/env python

'''
Evaluation scipt for Spanet architecture. 

'''

# python packages
import torch
import argparse
import os
import uproot
import gc
import numpy as np
from glob import glob
import sys
import h5py
import awkward as ak

# SOME HACK ON mac
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# spanet
from spanet import Options
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.dataset.event_info import EventInfo

# global variables
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
log = logging.getLogger('evaluate')

# multiprocessing
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass



def main():

    # user options
    ops = options() 
    
    # decide on device --> NOT CURRENTLY USED
    # device = ops.device 
    # if not device:
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # check if multiprocessing should be done
    input_files = handleInput(ops.inFile)

    # get list of ttrees using the first input file
    treeNames = ["trees_SRRPV_"]
    if ops.doSystematics:
        with uproot.open(input_files[0]) as f:
            treeNames = [i.split(";")[0] for i in f.keys() if "trees" in i]
        # remove large R systematics
        treeNames = [i for i in treeNames if "R10" not in i]
        log.info(f"Running over {len(treeNames)} trees including systematics")

    # make output dir
    if not os.path.isdir(ops.outDir):
        os.makedirs(ops.outDir)

    # create evaluation job dictionaries
    config  = []
    for inFileName in input_files:
        
        # create outfile tag
        tag = f"minJetPt{ops.minJetPt}_maxNjets{ops.maxNjets}_v{ops.version}"
        config.append({
            "inFileName" : inFileName,
            "treeNames" : treeNames,
            "tag" : tag
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
    parser.add_argument("-i", "--inFile", help="Data file to evaluate on.", default=None, required=True)
    parser.add_argument("-o", "--outDir", help="Directory to save evaluation output to.", default="./")
    # parser.add_argument("-d",  "--device", help="Device to use.", default=None)
    parser.add_argument("-j", "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("-l", "--log_directory", help="Pretrained weights to evaluate with.", default=None, required=True)
    parser.add_argument('-v', "--version", default="0", help="Production version")
    parser.add_argument('--minJetPt', default=50, type=int, help="Minimum selected jet pt")
    parser.add_argument('--maxNjets', default=8, type=int, help="Maximum number of leading jets retained in h5 files")
    parser.add_argument('--doSystematics', action="store_true", help="Create h5 files for systematic trees.")
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files.")
    return parser.parse_args()

def handleInput(data):
    # otherwise return 
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".root" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted(os.listdir(data))
    elif "*" in data:
        return sorted(glob(data))
    return []

def append_jet_selection(original, new):
    return np.concatenate([original, np.expand_dims(np.logical_and(original[:,:,-1],new),-1)],-1)

def evaluate(config):
    ''' perform the full model evaluation '''

    log.info(f"evaluating on {config['inFileName']}")
    ops = options()

    # make output file name
    outFileName = os.path.join(ops.outDir, os.path.basename(config["inFileName"])).replace(".root",f"_{config['tag']}_spanet.h5")
    if os.path.isfile(outFileName) and not ops.doOverwrite:
        log.info(f"File already exists not evaluating on: {outFileName}")
        return

    # Load the options that were used for this run and set the testing-dataset value
    model_options = Options.load(f"{ops.log_directory}/options.json")
    # get from event info
    event_info = EventInfo.read_from_ini(model_options.event_info_file)

    # Create model and disable all training operations for speed
    model = JetReconstructionNetwork(model_options)

    # Load the best-performing checkpoint on validation data
    checkpoint = torch.load(sorted(glob(f"{ops.log_directory}/checkpoints/epoch*"))[-1], map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint)

    # evaluate
    outData = {}
    model.eval()
    with torch.no_grad():
        for treeName in config["treeNames"]:
            log.info(f"Evaluationg {treeName}")

            # load data
            with uproot.open(config["inFileName"]) as f:
                tree = f[treeName]

                # pick up kinematics
                kinem = {}
                for key in ["e","pt","eta","phi"]:
                    kinem[key] = loadBranchAndPad(tree[f"jet_{key}"], ops.maxNjets) # need to apply njet, jet pt cuts --> cuts is the biggest anoyance here
                
                jet_selection = np.expand_dims(np.ones(kinem["pt"].shape),-1)
                jet_selection = append_jet_selection(jet_selection, kinem["pt"] >= ops.minJetPt)
                # apply final jet selection
                jet_selection = jet_selection.astype(bool)
                for key in kinem.keys():
                    kinem[key][~jet_selection[:,:,-1]] = 0

                # compute mass
                kinem["px"] = kinem["pt"] * np.cos(kinem["phi"])
                kinem["py"] = kinem["pt"] * np.sin(kinem["phi"])
                kinem["pz"] = kinem["pt"] * np.sinh(kinem["eta"])
                # negatives can happen from slight numerical impression I think
                kinem["mass"] =  np.nan_to_num(np.sqrt(kinem["e"]**2 - kinem["px"]**2 - kinem["py"]**2 - kinem["pz"]**2),0)

                # convert to tensors
                for key, val in kinem.items():
                    kinem[key] = torch.Tensor(val)

                # construct source
                source_mask = (kinem["pt"] != 0).bool()
                source_data = {}
                features = []

                # log data if desired
                for index, (feature, normalize, log_transform) in enumerate(event_info.source_features):
                    source_data[feature] = kinem[feature]
                    features.append(feature)
                    if log_transform:
                        source_data[feature] = torch.log(torch.clamp(source_data[feature], min=1e-6)) * source_mask
                    if normalize:
                        mean = float(getattr(model_options,f"{feature}_mean"))
                        std = float(getattr(model_options,f"{feature}_std"))
                        log.debug(f"{feature} mean {mean}, std {std}")
                        source_data[feature] = (source_data[feature] - mean) / std * source_mask

                # make input data
                source_data = torch.stack([source_data[feature] for feature in features],-1)
                log.debug(f"Source data {source_data.shape}, mask {source_mask.shape}")

                # prepare four momenta to be used with predictions
                mom = torch.stack([kinem[key] for key in ["e","px","py","pz"]],-1)

            if source_data.shape[0] == 0:
                log.info(f"File has no events after selections: {config['inFileName']}")
                return
            
            N = source_data.shape[0]
            predictions, classifications = model.predict_jets_and_particles(source_data=source_data[:N], source_mask=source_mask[:N])
            predictions = np.stack(predictions,1)
            log.debug(f"Predictions {predictions.shape}, Four-momenta {mom.shape}")

            # get masses
            mom_temp = np.expand_dims(mom,1)[:N] # go to (nEvents,1,nJets,4-mom)
            predictions_temp = np.repeat(np.expand_dims(predictions,-1),mom.shape[-1],-1) # go to (nEvents, nGluinos, nGluinoChildre, 4-mom)
            log.debug(f"After reshapes: Predictions {predictions.shape}, Four-momenta {mom.shape}")
            m = np.take_along_axis(mom_temp,predictions_temp,2).sum(2) # take along nJet axis and sum along axis to get (nEvents, nGluino, 4-mom)
            m = np.sqrt(m[:,:,0]**2 - m[:,:,1]**2 - m[:,:,2]**2 - m[:,:,3]**2) # compute mass
            del mom_temp, predictions_temp
            gc.collect()

            # save to dictionary
            outData[treeName] = {"predictions" : predictions, "mass_pred" : m}
    
    # save options # TO-DO: remove after training updated
    #model_options.target_symmetries = str(model_options.target_symmetries)
    #model_options.save("test.json")

    # save final file
    log.info(f"Saving to {outFileName}")
    with h5py.File(outFileName, 'w') as hf:
        for key, val in outData.items():
            Group = hf.create_group(key)
            for k, v in val.items():
                log.debug(f"    {key}/{k} {v.shape}")
                Group.create_dataset(k,data=v)
    log.info("Done!")

def loadBranchAndPad(branch, maxNjets):
    a = branch.array()
    a = ak.to_numpy(ak.fill_none(ak.pad_none(a, max(maxNjets,np.max(ak.num(a)))),0))
    return a
    
if __name__ == "__main__":
    main()
