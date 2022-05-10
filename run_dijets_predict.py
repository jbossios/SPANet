import os

VERSION = 60 # trained with all masses UDS+UDB full+partial events and using max8jets
PATH = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_expanded/python/'
OUT_PATH = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_Predictions/Dijets/'
SLICE = 'JZ12'

###################################
# DO NOT MODIFY (below this line)
###################################

rtags = ['r9364', 'r10201', 'r10724']

PATH += f'{SLICE}/'

# Create output folder
output_path = f'{OUT_PATH}v{VERSION}'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loop over slices
commands = []
for file_name in os.listdir(PATH):
    if 'spanet.h5' not in file_name: continue # skip other formats for other networks
    splits = file_name.split('.')
    dsid = splits[2]
    rtag = [tag for item in splits for tag in rtags if tag in item][0]
    extension = '.'.join(splits[4:6])
    if 'spanet.h5' not in file_name: continue # skip other formats for other networks
    commands.append(f'python3 predict.py  ./spanet_output/version_{VERSION} {output_path}/dijets_v{VERSION}_output_{dsid}_{rtag}_{extension}.h5 -tf {PATH}{file_name} --gpu')
command = ' && '.join(commands)+' &'
os.system(command)
