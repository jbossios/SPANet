
#!/usr/bin/env python

PATH    = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/Katib_HyperparameterOptimization/Outputs/'
Version = 'v18image-v01'
outPATH = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/Katib_HyperparameterOptimization/H5Files/'
TestFile  = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_inputs/signal_UDB_UDS_testing_v40.h5'

###################################
# DO NOT MODIFY (below this line)
###################################

PATH = PATH + Version + '/'

EventFile = 'event_files/signal.ini'

import os,sys

# Create output folder
if not os.path.exists('{}{}'.format(outPATH,Version)):
  os.makedirs('{}{}'.format(outPATH,Version))

commands = []

# Loop over files/folders in PATH
for Folder in os.listdir(PATH):
  if 'Model' not in Folder: continue # skip undesired folder/file
  ID        = Folder.split('Model_0.')[1]
  output_file = 'signal_testing_{}_output.h5'.format(ID)
  if os.path.exists(output_file): continue # skip ID for which there is already a prediction
  inputPATH = '{}{}/spanet_output/version_0'.format(PATH,Folder)
  commands.append('python3 predict.py {} {}{}/signal_testing_{}_output.h5 -tf {} -ef {}'.format(inputPATH,outPATH,Version,ID,TestFile,EventFile))

# Run commands
command = ' && '.join(commands)
command += ' &'
print(command)
os.system(command)
