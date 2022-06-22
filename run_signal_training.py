import os
import sys

if __name__ == '__main__':
  """ Submit SPANet training or evaluation jobs """
  # Settings
  run_type = 'train' # options: train, predict
  path = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_OptionFiles/'
  versions = [i for i in range(96, 97)]  # network settings
  # create dummy output folder to force the output version that I want
  if run_type == 'train' and not os.path.exists(f'spanet_output/version_{versions[0]-1}'):
    os.makedirs(f'spanet_output/version_{versions[0]-1}')
  commands = []
  for version in versions:
    inpath = f'{path}v{version}/'
    for infile in os.listdir(inpath):
      if run_type == 'train':
        command = f'python3 train.py -of options_files/{infile} --random_seed 1 --gpus 1 > Log_v{version}_{run_type} 2>&1'
      elif run_type == 'predict':
        with open(f'options_files/{infile}', 'r') as opt:
          lines = opt.readlines()
          for line in lines:
            if 'training' in line and 'h5' in line:
              h5_file = line.split(':')[1].replace(',' ,'').replace('"', '').replace('\n', '')
        command = f'python3 predict.py ./spanet_output/version_{version} ./signal_full_v{version}_output.h5 -tf{h5_file} --gpu > Log_v{version}_{run_type} 2>&1'
      else:
        print(F'ERROR: {run_type} not implemented yet, exiting')
        sys.exit(1)
      commands.append(command)
  command = ' && '.join(commands)
  command += ' &'
  print(command)
  os.system(command)
