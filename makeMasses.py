import h5py
import numpy as np
import matplotlib.pyplot as plt

# prepare e,px,py,pz
xFile = "user.abadea.512946.e8448_e7400_s3126_r10724_r10726_p5083.30449056._000001.trees_SRRPV_minJetPt20_minNjets10_maxNjets15_RDR_dr_v01.h5"
with h5py.File(xFile, "r") as hf:
	x = np.stack([np.array(hf[f'source/{key}']) for key in ["pt","eta","phi","mass"]],-1)

px = x[:,:,0] * np.cos(x[:,:,2])
py = x[:,:,0] * np.sin(x[:,:,2])
pz = x[:,:,0] * np.sinh(x[:,:,1])
e = np.sqrt(x[:,:,3]**2 + px**2 + py**2 + pz**2) # m^2 + p^2
x = np.stack([e,px,py,pz],-1)
print(x.shape)

# load labels
yFile = "user.abadea.512946.e8448_e7400_s3126_r10724_r10726_p5083.30449056._000001.trees_SRRPV_minJetPt20_minNjets10_maxNjets15_RDR_dr_v01_spanet.h5"
with h5py.File(yFile, "r") as hf:
	g1 = np.stack([np.array(hf[f'g1/q{key}']) for key in range(1,6)],-1)
	g2 = np.stack([np.array(hf[f'g2/q{key}']) for key in range(1,6)],-1)
	print(g1.shape, g2.shape)

# pickup correct jets and sum
g1m = np.take_along_axis(x, np.expand_dims(g1,-1).repeat(4,-1), 1).sum(1)
g2m = np.take_along_axis(x, np.expand_dims(g2,-1).repeat(4,-1), 1).sum(1)
print(g1m.shape, g2m.shape)

# create masses
g1m = np.sqrt(g1m[:,0]**2 - g1m[:,1]**2 - g1m[:,2]**2 - g1m[:,3]**2)
g2m = np.sqrt(g2m[:,0]**2 - g2m[:,1]**2 - g2m[:,2]**2 - g2m[:,3]**2)
print(g1m.shape, g2m.shape)

# plot
bins = np.linspace(0, 5000, 50)
plt.hist(g1m, bins=bins, histtype="step", color="blue", label="gluino 1")
plt.hist(g2m, bins=bins, histtype="step", color="red", label="gluino 2")
plt.legend()
plt.show()