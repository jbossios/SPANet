import kfp
import sys

def submit_pipeline(date, spanet_version, pipeline_id, path, experiment_id = 'dijetseval11032022'):
  client = kfp.Client()
  pipeline_name = f'spanet-dijets-eval-{date}-{spanet_version}-id{pipeline_id}'
  pipeline_file = f'{path}spanet_dijets_eval_{date}_{spanet_version}_{pipeline_id}.yaml'
  print(f'INFO: Submitting {pipeline_file}...')
  if client.get_pipeline_id(pipeline_name): # pipeline already exists
    print(f'ERROR: Pipeline already exists. Delete pipeline with the following:')
    print('import kfp')
    print('client = kfp.Client()')
    print(f'client.delete_pipeline(client.get_pipeline_id("{pipeline_name}"))')
    sys.exit(1)
  client.upload_pipeline(pipeline_file, pipeline_name)
  exp = client.get_experiment(experiment_name=experiment_id)
  run = client.run_pipeline(exp.id, pipeline_name, pipeline_file)

if __name__ == '__main__':
  date = '11032022'
  n_yaml_files_per_version = 4
  versions = [f'v{i}' for i in range(74, 92)]
  path_to_yaml_files = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_Pipelines_yaml_files/'
  for version in versions:
    for i in range(n_yaml_files_per_version):
      submit_pipeline(date, version, i, path_to_yaml_files)
  print('>>> ALL DONE <<<')
