import jinja2
import yaml
from common import yaml_utils
import os
from common import utils

config_filename = "test-exp.yaml"
config = yaml_utils.read(config_filename)

RESOURCES_DIR = os.path.join(utils.ROOT_DIR, 'experiment', 'resources')

JINJA_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
     loader=jinja2.FileSystemLoader(RESOURCES_DIR),
)

template = JINJA_ENV.get_template('runner-startup-script-template.sh')

instance_name = "foo"
benchmark = "bloaty"
experiment = "test"
fuzzer = "afl"
trial_id = "0"
fuzz_target = "also bloaty?"
local_experiment = True
docker_image_url = "https://not-real.url"

experiment_config = config
experiment_config["no_seeds"] = False
experiment_config["no_dictionaries"] = True
experiment_config["oss_fuzz_corpus"] = False
experiment_config["runner_num_cpu_cores"] = 1

kwargs = {
         'instance_name': instance_name,
         'benchmark': benchmark,
         'experiment': experiment,
         'fuzzer': fuzzer,
         'trial_id': trial_id,
         'max_total_time': experiment_config['max_total_time'],
         'experiment_filestore': experiment_config['experiment_filestore'],
         'report_filestore': experiment_config['report_filestore'],
         'fuzz_target': fuzz_target,
         'docker_image_url': docker_image_url,
         'docker_registry': experiment_config['docker_registry'],
         'local_experiment': local_experiment,
         'no_seeds': experiment_config['no_seeds'],
         'no_dictionaries': experiment_config['no_dictionaries'],
         'oss_fuzz_corpus': experiment_config['oss_fuzz_corpus'],
         'num_cpu_cores': experiment_config['runner_num_cpu_cores'],
         'seeds_per_trial_dir': experiment_config['seeds_per_trial_dir'],
         'use_seeds_per_trial': experiment_config['use_seeds_per_trial'],
}

print(template.render(**kwargs))
