import os
import sys

if __name__ == '__main__':
  """ Submit SPANet training or evaluation jobs """
  # Settings
  run_type = 'train' # options: train, predict
  path = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_OptionFiles/'
  training_file = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/GG_rpv/PROD0/h5/v0/user.abadea.GGrpv2x3ALL_minJetPt50_minNjets6_maxNjets8_RDR_dr_v0.h5'
  root_file = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/GG_rpv/PROD2/user.abadea.mc16_13TeV.504518.MGPy8EG_A14NNPDF23LO_GG_rpv_UDB_1400_squarks.e8258_s3126_r10724_p5083.PROD2_trees.root/user.abadea.504518.e8258_e7400_s3126_r10724_r10726_p5083.29777484._000002.trees.root'
  versions = [i for i in range(97, 98)]  # network settings
  # create dummy output folder to force the output version that I want
  if run_type == 'train' and not os.path.exists(f'spanet_output/version_{versions[0]-1}'):
    os.makedirs(f'spanet_output/version_{versions[0]-1}')
  commands = []
  for version in versions:
    opt_infile = f'signal_v{version}_partial.json'
    if run_type == 'train':
      command = f'python3 train_test.py -of options_files/{opt_infile} -ef event_files/signal.ini -tf {training_file} --random_seed 1 --gpus 1 > Log_v{version}_{run_type} 2>&1'
    elif run_type == 'predict':
      command = f'python3 evaluate.py -l ./spanet_output/version_{version} -v {version} -i {root_file} > Log_v{version}_{run_type} 2>&1'
    else:
      print(F'ERROR: {run_type} not implemented yet, exiting')
      sys.exit(1)
    commands.append(command)
  command = ' && '.join(commands)
  command += ' &'
  print(command)
  os.system(command)
