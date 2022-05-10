import os

if __name__ == '__main__':
  """ Copy SPANet predictions (i.e. H5 files) to EOS"""
  versions = [f'v{i}' for i in range(72, 92)]
  path = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_Predictions/Signal/'
  for version in versions:
    if not os.path.exists(f'{path}{version}/'):
      os.makedirs(f'{path}{version}/')
    os.system(f'cp signal_full_{version}_output.h5 {path}{version}/')
