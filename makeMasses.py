import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
parser.add_argument("-x", "--xFile", help="Data file.", default=None, required=True)
parser.add_argument("-y", "--yFile", help="Prediction file.", default=None, required=True)
ops = parser.parse_args()

# prepare e,px,py,pz
xFile = "user.abadea.512946.e8448_e7400_s3126_r10724_r10726_p5083.30449056._000001.trees_SRRPV_minJetPt20_minNjets10_maxNjets15_RDR_dr_v01.h5"
with h5py.File(ops.xFile, "r") as hf:
	x = np.stack([np.array(hf[f'source/{key}']) for key in ["pt","eta","phi","mass"]],-1)

px = x[:,:,0] * np.cos(x[:,:,2])
py = x[:,:,0] * np.sin(x[:,:,2])
pz = x[:,:,0] * np.sinh(x[:,:,1])
e = np.sqrt(x[:,:,3]**2 + px**2 + py**2 + pz**2) # m^2 + p^2
x = np.stack([e,px,py,pz],-1)
print(x.shape)

# load labels
yFile = "user.abadea.512946.e8448_e7400_s3126_r10724_r10726_p5083.30449056._000001.trees_SRRPV_minJetPt20_minNjets10_maxNjets15_RDR_dr_v01_spanet.h5"
with h5py.File(ops.yFile, "r") as hf:
	g1 = np.stack([np.array(hf[f'g1/q{key}']) for key in range(1,6)],-1)
	g2 = np.stack([np.array(hf[f'g2/q{key}']) for key in range(1,6)],-1)
	print(g1.shape, g2.shape)

# pickup correct jets and sum
g1p = np.take_along_axis(x, np.expand_dims(g1,-1).repeat(4,-1), 1)
g2p = np.take_along_axis(x, np.expand_dims(g2,-1).repeat(4,-1), 1)

# create gluino masses
g1m = g1p.sum(1)
g2m = g2p.sum(1)
g1m = np.sqrt(g1m[:,0]**2 - g1m[:,1]**2 - g1m[:,2]**2 - g1m[:,3]**2)
g2m = np.sqrt(g2m[:,0]**2 - g2m[:,1]**2 - g2m[:,2]**2 - g2m[:,3]**2)
print(g1m.shape, g2m.shape)

# create neutralino masses
neum = np.stack([g1p[:,[2,3,4]],g2p[:,[2,3,4]]],1).sum(2)
neum = np.sqrt(neum[:,:,0]**2 - neum[:,:,1]**2 - neum[:,:,2]**2 - neum[:,:,3]**2)
print(neum.shape)

# plot
bins = np.linspace(0, 3500, 50)
plt.hist(g1m, bins=bins, histtype="step", color="blue", label="gluino 1")
plt.hist(g2m, bins=bins, histtype="step", color="red", label="gluino 2")
plt.hist(neum[:,0].flatten(), bins=bins, histtype="step", color="blue", label="neutralino 1", ls="--")
plt.hist(neum[:,1].flatten(), bins=bins, histtype="step", color="red", label="neutralino 2", ls="--")
plt.legend()
plt.ylabel("Number of Objects")
plt.xlabel("Mass [GeV]")
plt.show()
