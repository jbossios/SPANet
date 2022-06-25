import uproot
import gc
import numpy as np
from glob import glob
import sys
import torch

# spanet
from spanet import Options
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.dataset.event_info import EventInfo

# from cannonball
sys.path.append("/Users/anthonybadea/Documents/ATLAS/rpvmj/cannonball")
from batcher import loadBranchAndPad

# load options
log_directory = "version_96"

# Load the options that were used for this run and set the testing-dataset value
options = Options.load(f"{log_directory}/options.json")
# get from event info
event_info = EventInfo.read_from_ini(options.event_info_file)
options.target_symmetries = event_info.mapped_targets.items()
options.num_features = event_info.num_features

# Create model and disable all training operations for speed
model = JetReconstructionNetwork(options)

# Load the best-performing checkpoint on validation data
checkpoint = torch.load(sorted(glob(f"{log_directory}/checkpoints/epoch*"))[-1], map_location='cpu')["state_dict"]
# hack while removing unused variables
ckpt = {}
for param_tensor in model.state_dict():
	ckpt[param_tensor] = checkpoint[param_tensor]
model.load_state_dict(ckpt) # eventually just load the checkpoint normally

# turn of gradients
for parameter in model.parameters():
    parameter.requires_grad_(False)

# get data in format
inFile = "../inputs/user.abadea.mc16_13TeV.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW.e7142_s3126_r9364_p5083.PROD3_trees.root/user.abadea.364704.e7142_e5984_s3126_r9364_r9315_p5083.29328273._000001.trees.root"
treeName = "trees_SRRPV_"
maxNjets = 8

# load data
with uproot.open(inFile) as f:
	tree = f[treeName]

	# pick up kinematics
	kinem = {}
	for key in ["e","pt","eta","phi"]:
		kinem[key] = loadBranchAndPad(tree[f"jet_{key}"], maxNjets)[:2] # need to apply njet, jet pt cuts --> cuts is the biggest anoyance here

	# compute mass
	kinem["px"] = kinem["pt"] * np.cos(kinem["phi"])
	kinem["py"] = kinem["pt"] * np.sin(kinem["phi"])
	kinem["pz"] = kinem["pt"] * np.sinh(kinem["eta"])
	# negatives can happen from slight numerical impression I think
	kinem["mass"] =  np.nan_to_num(np.sqrt(kinem["e"]**2 - kinem["px"]**2 - kinem["py"]**2 - kinem["pz"]**2),0)

	# convert to tensors
	for key, val in kinem.items():
		kinem[key] = torch.Tensor(val)

	# construct sort
	source_mask = (kinem["pt"] != 0).bool()
	source_data = {key : kinem[key] for key in ["mass","pt","eta","phi"]} # copy in order to normalize before evaluation

	# log data if desired
	for index, (feature, normalize, log_transform) in enumerate(event_info.source_features):
		if log_transform:
			source_data[feature] = torch.log(torch.clamp(source_data[feature], min=1e-6)) * source_mask
		if normalize: # only include nonzero elements in mean, std since 0 indicates padded
			mean = source_data[feature][source_data[feature]!=0].mean()
			std = source_data[feature][source_data[feature]!=0].std()
			source_data[feature] = (source_data[feature] - mean) / std * source_mask

	# make input data
	source_data = torch.stack([
		source_data["mass"],
		source_data["pt"],
		source_data["eta"],
		source_data["phi"]
	],-1)

	print(f"Source data {source_data.shape}, mask {source_mask.shape}")


# evaluate
model.eval()
with torch.no_grad():

	predictions, classifications = model.predict_jets_and_particles(source_data=source_data, source_mask=source_mask)
	predictions = np.stack(predictions)
	print(type(predictions), predictions.shape) 
	print(predictions)
	print(type(classifications), classifications.shape)
	print(classifications)
	
	# convert to masses
	mom = torch.stack([
		kinem["e"],
		kinem["px"],
		kinem["py"],
		kinem["pz"]
	],-1)
	mom = np.expand_dims(mom,1) # go to (nEvents,1,nJets,4-mom)
	predictions = np.repeat(np.expand_dims(predictions,-1),mom.shape[-1],-1) # go to (nEvents, nGluinos, nGluinoChildre, 4-mom)
	m = np.take_along_axis(mom,predictions,2).sum(2) # take along nJet axis and sum along axis to get (nEvents, nGluino, 4-mom)
	m = np.sqrt(m[:,:,0]**2 - m[:,:,1]**2 - m[:,:,2]**2 - m[:,:,3]**2) # compute mass

	# To-do: why does the mass keep changing on each evaluation?
	print(m.shape)
	print(m)

