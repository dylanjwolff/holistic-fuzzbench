#!/usr/bin/env python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Creates a dispatcher VM in GCP and sends it all the files and configurations
it needs to begin an experiment."""

import argparse
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import namedtuple
from typing import Dict, List, Optional, Union

import jinja2
import yaml

from common import benchmark_utils
from common import experiment_utils
from common import filestore_utils
from common import filesystem
from common import fuzzer_utils
from common import gcloud
from common import gsutil
from common import logs
from common import new_process
from common import utils
from common import yaml_utils

BENCHMARKS_DIR = os.path.join(utils.ROOT_DIR, 'benchmarks')
FUZZERS_DIR = os.path.join(utils.ROOT_DIR, 'fuzzers')
RESOURCES_DIR = os.path.join(utils.ROOT_DIR, 'experiment', 'resources')
FUZZER_NAME_REGEX = re.compile(r'^[a-z][a-z0-9_]+$')
EXPERIMENT_CONFIG_REGEX = re.compile(r'^[a-z0-9-]{0,30}$')
FILTER_SOURCE_REGEX = re.compile(r'('
                                 r'^\.git/|'
                                 r'^\.pytype/|'
                                 r'^\.venv/|'
                                 r'^.*\.pyc$|'
                                 r'^__pycache__/|'
                                 r'.*~$|'
                                 r'\#*\#$|'
                                 r'\.pytest_cache/|'
                                 r'.*/test_data/|'
                                 r'^docker/generated.mk$|'
                                 r'^docs/)')
_OSS_FUZZ_CORPUS_BACKUP_URL_FORMAT = (
    'gs://{project}-backup.clusterfuzz-external.appspot.com/corpus/'
    'libFuzzer/{fuzz_target}/public.zip')
DEFAULT_CONCURRENT_BUILDS = 30

Requirement = namedtuple('Requirement',
                         ['mandatory', 'type', 'lowercase', 'startswith'])


def _set_default_config_values(config: Dict[str, Union[int, str, bool]],
                               local_experiment: bool):
    """Set the default configuration values if they are not specified."""
    config['local_experiment'] = local_experiment
    config['worker_pool_name'] = config.get('worker_pool_name', '')
    config['snapshot_period'] = config.get(
        'snapshot_period', experiment_utils.DEFAULT_SNAPSHOT_SECONDS)
    config['private'] = config.get('private', False)


def _validate_config_parameters(
        config: Dict[str, Union[int, str, bool]],
        config_requirements: Dict[str, Requirement]) -> bool:
    """Validates if the required |params| exist in |config|."""
    if 'cloud_experiment_bucket' in config or 'cloud_web_bucket' in config:
        logs.error('"cloud_experiment_bucket" and "cloud_web_bucket" are now '
                   '"experiment_filestore" and "report_filestore".')

    missing_params, optional_params = [], []
    for param, requirement in config_requirements.items():
        if param in config:
            continue
        if requirement.mandatory:
            missing_params.append(param)
            continue
        optional_params.append(param)

    for param in missing_params:
        logs.error('Config does not contain required parameter "%s".', param)

    return not missing_params


# pylint: disable=too-many-arguments
def _validate_config_values(
        config: Dict[str, Union[str, int, bool]],
        config_requirements: Dict[str, Requirement]) -> bool:
    """Validates if |params| types and formats in |config| are correct."""

    valid = True
    for param, value in config.items():
        requirement = config_requirements.get(param, None)
        # Unrecognised parameter.
        error_param = 'Config parameter "%s" is "%s".'
        if requirement is None:
            valid = False
            error_reason = 'This parameter is not recognized.'
            logs.error(f'{error_param} {error_reason}', param, str(value))
            continue

        if not isinstance(value, requirement.type):
            valid = False
            error_reason = f'It must be a {requirement.type}.'
            logs.error(f'{error_param} {error_reason}', param, str(value))

        if not isinstance(value, str):
            continue

        if requirement.lowercase and not value.islower():
            valid = False
            error_reason = 'It must be a lowercase string.'
            logs.error(f'{error_param} {error_reason}', param, str(value))

        if requirement.startswith and not value.startswith(
                requirement.startswith):
            valid = False
            error_reason = (
                'Local experiments only support Posix file systems filestores.'
                if config.get('local_experiment', False) else
                'Google Cloud experiments must start with "gs://".')
            logs.error(f'{error_param} {error_reason}', param, value)

    return valid


# pylint: disable=too-many-locals
def read_and_validate_experiment_config(config_filename: str) -> Dict:
    """Reads |config_filename|, validates it, finds as many errors as possible,
    and returns it."""
    # Reads config from file.
    config = yaml_utils.read(config_filename)

    # Validates config contains all the required parameters.
    local_experiment = config.get('local_experiment', False)

    # Requirement of each config field.
    config_requirements = {
        'experiment_filestore':
            Requirement(True, str, True, '/' if local_experiment else 'gs://'),
        'report_filestore':
            Requirement(True, str, True, '/' if local_experiment else 'gs://'),
        'docker_registry':
            Requirement(True, str, True, ''),
        'trials':
            Requirement(True, int, False, ''),
        'max_total_time':
            Requirement(True, int, False, ''),
        'cloud_compute_zone':
            Requirement(not local_experiment, str, True, ''),
        'cloud_project':
            Requirement(not local_experiment, str, True, ''),
        'worker_pool_name':
            Requirement(not local_experiment, str, False, ''),
        'cloud_sql_instance_connection_name':
            Requirement(False, str, True, ''),
        'snapshot_period':
            Requirement(False, int, False, ''),
        'local_experiment':
            Requirement(False, bool, False, ''),
        'private':
            Requirement(False, bool, False, ''),
        'merge_with_nonprivate':
            Requirement(False, bool, False, ''),
        'preemptible_runners':
            Requirement(False, bool, False, ''),
        'runner_machine_type':
            Requirement(False, str, True, ''),
        'runner_num_cpu_cores':
            Requirement(False, int, False, ''),
        'use_seed_sampling':
            Requirement(False, bool, False, ''),
        'seed_sampling_randomness_init':
            Requirement(False, int, False, ''),
        'seed_sampling_distribution':
            Requirement(False, str, False, ''),
        'seed_sampling_mean_utilization':
            Requirement(False, float, False, ''),
        'runner_memory':
            Requirement(False, str, False, ''),
    }

    all_params_valid = _validate_config_parameters(config, config_requirements)
    all_values_valid = _validate_config_values(config, config_requirements)
    if not all_params_valid or not all_values_valid:
        raise ValidationError(f'Config: {config_filename} is invalid.')

    _set_default_config_values(config, local_experiment)
    return config


class ValidationError(Exception):
    """Error validating user input to this program."""


def get_directories(parent_dir):
    """Returns a list of subdirectories in |parent_dir|."""
    return [
        directory for directory in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, directory))
    ]


# pylint: disable=too-many-locals
def validate_custom_seed_corpus(custom_seed_corpus_dir, benchmarks):
    """Validate seed corpus provided by user"""
    if not os.path.isdir(custom_seed_corpus_dir):
        raise ValidationError(
            f'Corpus location "{custom_seed_corpus_dir}" is invalid.')

    for benchmark in benchmarks:
        benchmark_corpus_dir = os.path.join(custom_seed_corpus_dir, benchmark)
        if not os.path.exists(benchmark_corpus_dir):
            raise ValidationError('Custom seed corpus directory for '
                                  f'benchmark "{benchmark}" does not exist.')
        if not os.path.isdir(benchmark_corpus_dir):
            raise ValidationError(
                f'Seed corpus of benchmark "{benchmark}" must be a directory.')
        if not os.listdir(benchmark_corpus_dir):
            raise ValidationError(
                f'Seed corpus of benchmark "{benchmark}" is empty.')


def validate_benchmarks(benchmarks: List[str]):
    """Parses and validates list of benchmarks."""
    benchmark_types = set()
    for benchmark in set(benchmarks):
        if benchmarks.count(benchmark) > 1:
            raise ValidationError(
                f'Benchmark "{benchmark}" is included more than once.')
        # Validate benchmarks here. It's possible someone might run an
        # experiment without going through presubmit. Better to catch an invalid
        # benchmark than see it in production.
        if not benchmark_utils.validate(benchmark):
            raise ValidationError(f'Benchmark "{benchmark}" is invalid.')

        benchmark_types.add(benchmark_utils.get_type(benchmark))

    if (benchmark_utils.BenchmarkType.CODE.value in benchmark_types and
            benchmark_utils.BenchmarkType.BUG.value in benchmark_types):
        raise ValidationError(
            'Cannot mix bug benchmarks with code coverage benchmarks.')


def validate_fuzzer(fuzzer: str):
    """Parses and validates a fuzzer name."""
    if not fuzzer_utils.validate(fuzzer):
        raise ValidationError(f'Fuzzer: {fuzzer} is invalid.')


def validate_experiment_name(experiment_name: str):
    """Validate |experiment_name| so that it can be used in creating
    instances."""
    if not re.match(EXPERIMENT_CONFIG_REGEX, experiment_name):
        raise ValidationError(
            f'Experiment name "{experiment_name}" is invalid. '
            f'Must match: "{EXPERIMENT_CONFIG_REGEX.pattern}"')


def set_up_experiment_config_file(config):
    """Set up the config file that will actually be used in the
    experiment (not the one given to run_experiment.py)."""
    filesystem.recreate_directory(experiment_utils.CONFIG_DIR)
    experiment_config_filename = (
        experiment_utils.get_internal_experiment_config_relative_path())
    with open(experiment_config_filename, 'w',
              encoding='utf-8') as experiment_config_file:
        yaml.dump(config, experiment_config_file, default_flow_style=False)


def check_no_uncommitted_changes():
    """Make sure that there are no uncommitted changes."""
    if subprocess.check_output(['git', 'diff'], cwd=utils.ROOT_DIR):
        raise ValidationError('Local uncommitted changes found, exiting.')


def get_git_hash(allow_uncommitted_changes):
    """Return the git hash for the last commit in the local repo."""
    try:
        output = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=utils.ROOT_DIR)
        return output.strip().decode('utf-8')
    except subprocess.CalledProcessError as error:
        if not allow_uncommitted_changes:
            raise error
        return ''


def start_experiment(  # pylint: disable=too-many-arguments
        experiment_name: str,
        config_filename: str,
        benchmarks: List[str],
        fuzzers: List[str],
        description: Optional[str] = None,
        no_seeds: bool = False,
        no_dictionaries: bool = False,
        oss_fuzz_corpus: bool = False,
        allow_uncommitted_changes: bool = False,
        concurrent_builds: Optional[int] = DEFAULT_CONCURRENT_BUILDS,
        measurers_cpus: Optional[int] = None,
        runners_cpus: Optional[int] = None,
        region_coverage: bool = False,
        custom_seed_corpus_dir: Optional[str] = None):
    """Start a fuzzer benchmarking experiment."""
    if not allow_uncommitted_changes:
        check_no_uncommitted_changes()

    validate_experiment_name(experiment_name)
    validate_benchmarks(benchmarks)

    config = read_and_validate_experiment_config(config_filename)
    config['fuzzers'] = fuzzers
    config['benchmarks'] = benchmarks
    config['experiment'] = experiment_name
    config['git_hash'] = get_git_hash(allow_uncommitted_changes)
    config['no_seeds'] = no_seeds
    config['no_dictionaries'] = no_dictionaries
    config['oss_fuzz_corpus'] = oss_fuzz_corpus
    config['description'] = description
    config['concurrent_builds'] = concurrent_builds
    config['measurers_cpus'] = measurers_cpus
    config['runners_cpus'] = runners_cpus
    config['runner_machine_type'] = config.get('runner_machine_type',
                                               'n1-standard-1')
    config['runner_num_cpu_cores'] = config.get('runner_num_cpu_cores', 1)
    assert (runners_cpus is None or
            runners_cpus >= config['runner_num_cpu_cores'])
    # Note this is only used if runner_machine_type is None.
    # 12GB is just the amount that KLEE needs, use this default to make KLEE
    # experiments easier to run.
    config['runner_memory'] = config.get('runner_memory', '12GB')
    config['region_coverage'] = region_coverage

    config['custom_seed_corpus_dir'] = custom_seed_corpus_dir
    if config['custom_seed_corpus_dir']:
        validate_custom_seed_corpus(config['custom_seed_corpus_dir'],
                                    benchmarks)

    if config.get('seed_sampling_mean_utilization'):
        assert 0 < config['seed_sampling_mean_utilization'] < 1
    if config.get('seed_sampling_distribution'):
        assert config['seed_sampling_distribution'] in ['EXP', 'UNIFORM']

    return start_experiment_from_full_config(config)


def start_experiment_from_full_config(config):
    """Start a fuzzer benchmarking experiment from a full (internal) config."""

    set_up_experiment_config_file(config)

    # Make sure we can connect to database.
    local_experiment = config.get('local_experiment', False)
    if not local_experiment:
        if 'POSTGRES_PASSWORD' not in os.environ:
            raise ValidationError(
                'Must set POSTGRES_PASSWORD environment variable.')
        gcloud.set_default_project(config['cloud_project'])

    start_dispatcher(config, experiment_utils.CONFIG_DIR)


def start_dispatcher(config: Dict, config_dir: str):
    """Start the dispatcher instance and run the dispatcher code on it."""
    dispatcher = get_dispatcher(config)
    # Is dispatcher code being run manually (useful for debugging)?
    copy_resources_to_bucket(config_dir, config)
    if not os.getenv('MANUAL_EXPERIMENT'):
        dispatcher.start()


def add_oss_fuzz_corpus(benchmark, oss_fuzz_corpora_dir):
    """Add latest public corpus from OSS-Fuzz as the seed corpus for various
    fuzz targets."""
    project = benchmark_utils.get_project(benchmark)
    fuzz_target = benchmark_utils.get_fuzz_target(benchmark)

    if not fuzz_target.startswith(project):
        full_fuzz_target = f'{project}_{fuzz_target}'
    else:
        full_fuzz_target = fuzz_target

    src_corpus_url = _OSS_FUZZ_CORPUS_BACKUP_URL_FORMAT.format(
        project=project, fuzz_target=full_fuzz_target)
    dest_corpus_url = os.path.join(oss_fuzz_corpora_dir, f'{benchmark}.zip')
    gsutil.cp(src_corpus_url, dest_corpus_url, parallel=True, expect_zero=False)


def copy_resources_to_bucket(config_dir: str, config: Dict):
    """Copy resources the dispatcher will need for the experiment to the
    experiment_filestore."""

    def filter_file(tar_info):
        """Filter out unnecessary directories."""
        if FILTER_SOURCE_REGEX.match(tar_info.name):
            return None
        return tar_info

    # Set environment variables to use corresponding filestore_utils.
    os.environ['EXPERIMENT_FILESTORE'] = config['experiment_filestore']
    os.environ['EXPERIMENT'] = config['experiment']
    experiment_filestore_path = experiment_utils.get_experiment_filestore_path()

    base_destination = os.path.join(experiment_filestore_path, 'input')

    # Send the local source repository to the cloud for use by dispatcher.
    # Local changes to any file will propagate.
    source_archive = 'src.tar.gz'
    with tarfile.open(source_archive, 'w:gz') as tar:
        tar.add(utils.ROOT_DIR, arcname='', recursive=True, filter=filter_file)
    filestore_utils.cp(source_archive, base_destination + '/', parallel=True)
    os.remove(source_archive)

    # Send config files.
    destination = os.path.join(base_destination, 'config')
    filestore_utils.rsync(config_dir, destination, parallel=True)

    # If |oss_fuzz_corpus| flag is set, copy latest corpora from each benchmark
    # (if available) in our filestore bucket.
    if config['oss_fuzz_corpus']:
        oss_fuzz_corpora_dir = (
            experiment_utils.get_oss_fuzz_corpora_filestore_path())
        for benchmark in config['benchmarks']:
            add_oss_fuzz_corpus(benchmark, oss_fuzz_corpora_dir)

    if config['custom_seed_corpus_dir']:
        for benchmark in config['benchmarks']:
            benchmark_custom_corpus_dir = os.path.join(
                config['custom_seed_corpus_dir'], benchmark)
            filestore_utils.cp(
                benchmark_custom_corpus_dir,
                experiment_utils.get_custom_seed_corpora_filestore_path() + '/',
                recursive=True,
                parallel=True)


class BaseDispatcher:
    """Class representing the dispatcher."""

    def __init__(self, config: Dict):
        self.config = config
        self.instance_name = experiment_utils.get_dispatcher_instance_name(
            config['experiment'])

    def start(self):
        """Start the experiment on the dispatcher."""
        raise NotImplementedError


class LocalDispatcher(BaseDispatcher):
    """Class representing the local dispatcher."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.process = None

    def start(self):
        """Start the experiment on the dispatcher."""
        container_name = 'dispatcher-container'
        experiment_filestore_path = os.path.abspath(
            self.config['experiment_filestore'])
        filesystem.create_directory(experiment_filestore_path)
        sql_database_arg = (
            'SQL_DATABASE_URL=sqlite:///'
            f'{os.path.join(experiment_filestore_path, "local.db")}'
            '?check_same_thread=False')

        docker_registry = self.config['docker_registry']
        set_instance_name_arg = f'INSTANCE_NAME={self.instance_name}'
        set_experiment_arg = f'EXPERIMENT={self.config["experiment"]}'
        filestore = self.config['experiment_filestore']
        shared_experiment_filestore_arg = f'{filestore}:{filestore}'
        # TODO: (#484) Use config in function args or set as environment
        # variables.
        set_docker_registry_arg = f'DOCKER_REGISTRY={docker_registry}'
        set_experiment_filestore_arg = (
            f'EXPERIMENT_FILESTORE={self.config["experiment_filestore"]}')

        filestore = self.config['report_filestore']
        shared_report_filestore_arg = f'{filestore}:{filestore}'
        set_report_filestore_arg = f'REPORT_FILESTORE={filestore}'
        set_snapshot_period_arg = (
            f'SNAPSHOT_PERIOD={self.config["snapshot_period"]}')
        docker_image_url = f'{docker_registry}/dispatcher-image'
        set_concurrent_builds_arg = (
            f'CONCURRENT_BUILDS={self.config["concurrent_builds"]}')
        set_worker_pool_name_arg = (
            f'WORKER_POOL_NAME={self.config["worker_pool_name"]}')
        environment_args = [
            '-e',
            'LOCAL_EXPERIMENT=True',
            '-e',
            set_instance_name_arg,
            '-e',
            set_experiment_arg,
            '-e',
            sql_database_arg,
            '-e',
            set_experiment_filestore_arg,
            '-e',
            set_snapshot_period_arg,
            '-e',
            set_report_filestore_arg,
            '-e',
            set_docker_registry_arg,
            '-e',
            set_concurrent_builds_arg,
            '-e',
            set_worker_pool_name_arg,
        ]
        command = [
            'docker',
            'run',
            '-ti',
            '--rm',
            '-v',
            '/var/run/docker.sock:/var/run/docker.sock',
            '-v',
            shared_experiment_filestore_arg,
            '-v',
            shared_report_filestore_arg,
        ] + environment_args + [
            '--shm-size=2g',
            '--cap-add=SYS_PTRACE',
            '--cap-add=SYS_NICE',
            f'--name={container_name}',
            docker_image_url,
            '/bin/bash',
            '-c',
            'rsync -r '
            '"${EXPERIMENT_FILESTORE}/${EXPERIMENT}/input/" ${WORK} && '
            'mkdir ${WORK}/src && '
            'tar -xvzf ${WORK}/src.tar.gz -C ${WORK}/src && '
            'PYTHONPATH=${WORK}/src python3 '
            '${WORK}/src/experiment/dispatcher.py || '
            '/bin/bash'  # Open shell if experiment fails.
        ]
        logs.info('Starting dispatcher with container name: %s', container_name)
        return new_process.execute(command, write_to_stdout=True)


class GoogleCloudDispatcher(BaseDispatcher):
    """Class representing the dispatcher instance on Google Cloud."""

    def start(self):
        """Start the experiment on the dispatcher."""
        with tempfile.NamedTemporaryFile(dir=os.getcwd(),
                                         mode='w') as startup_script:
            self.write_startup_script(startup_script)
            if not gcloud.create_instance(self.instance_name,
                                          gcloud.InstanceType.DISPATCHER,
                                          self.config,
                                          startup_script=startup_script.name):
                raise RuntimeError('Failed to create dispatcher.')
            logs.info('Started dispatcher with instance name: %s',
                      self.instance_name)

    def _render_startup_script(self):
        """Renders the startup script template and returns the result as a
        string."""
        jinja_env = jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            loader=jinja2.FileSystemLoader(RESOURCES_DIR),
        )
        template = jinja_env.get_template(
            'dispatcher-startup-script-template.sh')
        cloud_sql_instance_connection_name = (
            self.config['cloud_sql_instance_connection_name'])

        kwargs = {
            'instance_name': self.instance_name,
            'postgres_password': os.environ['POSTGRES_PASSWORD'],
            'experiment': self.config['experiment'],
            'cloud_project': self.config['cloud_project'],
            'experiment_filestore': self.config['experiment_filestore'],
            'cloud_sql_instance_connection_name':
                (cloud_sql_instance_connection_name),
            'docker_registry': self.config['docker_registry'],
            'concurrent_builds': self.config['concurrent_builds'],
            'worker_pool_name': self.config['worker_pool_name'],
            'private': self.config['private'],
        }
        if 'worker_pool_name' in self.config:
            kwargs['worker_pool_name'] = self.config['worker_pool_name']
        return template.render(**kwargs)

    def write_startup_script(self, startup_script_file):
        """Get the startup script to start the experiment on the dispatcher."""
        startup_script = self._render_startup_script()
        startup_script_file.write(startup_script)
        startup_script_file.flush()


def get_dispatcher(config: Dict) -> BaseDispatcher:
    """Return a dispatcher object created from the right class (i.e. dispatcher
    factory)."""
    if config.get('local_experiment'):
        return LocalDispatcher(config)
    return GoogleCloudDispatcher(config)


def main():
    """Run an experiment."""
    return run_experiment_main()


def run_experiment_main(args=None):
    """Run an experiment."""
    logs.initialize()

    parser = argparse.ArgumentParser(
        description='Begin an experiment that evaluates fuzzers on one or '
        'more benchmarks.')

    all_benchmarks = benchmark_utils.get_all_benchmarks()
    coverage_benchmarks = benchmark_utils.get_coverage_benchmarks()
    parser.add_argument('-b',
                        '--benchmarks',
                        help=('Benchmark names. '
                              'All code coverage benchmarks of them by '
                              'default.'),
                        nargs='+',
                        required=False,
                        default=coverage_benchmarks,
                        choices=all_benchmarks)
    parser.add_argument('-c',
                        '--experiment-config',
                        help='Path to the experiment configuration yaml file.',
                        required=True)
    parser.add_argument('-e',
                        '--experiment-name',
                        help='Experiment name.',
                        required=True)
    parser.add_argument('-d',
                        '--description',
                        help='Description of the experiment.',
                        required=False)
    parser.add_argument('-cb',
                        '--concurrent-builds',
                        help='Max concurrent builds allowed.',
                        default=DEFAULT_CONCURRENT_BUILDS,
                        type=int,
                        required=False)
    parser.add_argument('-mc',
                        '--measurers-cpus',
                        help='Cpus available to the measurers.',
                        type=int,
                        required=False)
    parser.add_argument('-rc',
                        '--runners-cpus',
                        help='Cpus available to the runners.',
                        type=int,
                        required=False)
    parser.add_argument('-cs',
                        '--custom-seed-corpus-dir',
                        help='Path to the custom seed corpus',
                        required=False)

    all_fuzzers = fuzzer_utils.get_fuzzer_names()
    parser.add_argument('-f',
                        '--fuzzers',
                        help='Fuzzers to use.',
                        nargs='+',
                        required=False,
                        default=None,
                        choices=all_fuzzers)
    parser.add_argument('-ns',
                        '--no-seeds',
                        help='Should trials be conducted without seed corpora.',
                        required=False,
                        default=False,
                        action='store_true')
    parser.add_argument('-nd',
                        '--no-dictionaries',
                        help='Should trials be conducted without dictionaries.',
                        required=False,
                        default=False,
                        action='store_true')
    parser.add_argument('-a',
                        '--allow-uncommitted-changes',
                        help='Skip check that no uncommited changes made.',
                        required=False,
                        default=False,
                        action='store_true')
    parser.add_argument('-cr',
                        '--region-coverage',
                        help='Use region as coverage metric.',
                        required=False,
                        default=False,
                        action='store_true')
    parser.add_argument(
        '-o',
        '--oss-fuzz-corpus',
        help='Should trials be conducted with OSS-Fuzz corpus (if available).',
        required=False,
        default=False,
        action='store_true')
    args = parser.parse_args(args)
    fuzzers = args.fuzzers or all_fuzzers

    concurrent_builds = args.concurrent_builds
    if concurrent_builds is not None and concurrent_builds <= 0:
        parser.error('The concurrent build argument must be a positive number,'
                     f' received {concurrent_builds}.')

    runners_cpus = args.runners_cpus
    if runners_cpus is not None and runners_cpus <= 0:
        parser.error('The runners cpus argument must be a positive number,'
                     f' received {runners_cpus}.')

    measurers_cpus = args.measurers_cpus
    if measurers_cpus is not None and measurers_cpus <= 0:
        parser.error('The measurers cpus argument must be a positive number,'
                     f' received {measurers_cpus}.')

    if runners_cpus is None and measurers_cpus is not None:
        parser.error('With the measurers cpus argument (received '
                     f'{measurers_cpus}) you need to specify the runners cpus '
                     'argument too.')

    if (runners_cpus if runners_cpus else 0) + (measurers_cpus if measurers_cpus
                                                else 0) > os.cpu_count():
        parser.error(f'The sum of runners ({runners_cpus}) and measurers cpus '
                     f'({measurers_cpus}) is greater than the available cpu '
                     f'cores (os.cpu_count()).')

    if args.custom_seed_corpus_dir:
        if args.no_seeds:
            parser.error('Cannot enable options "custom_seed_corpus_dir" and '
                         '"no_seeds" at the same time')
        if args.oss_fuzz_corpus:
            parser.error('Cannot enable options "custom_seed_corpus_dir" and '
                         '"oss_fuzz_corpus" at the same time')

    start_experiment(args.experiment_name,
                     args.experiment_config,
                     args.benchmarks,
                     fuzzers,
                     description=args.description,
                     no_seeds=args.no_seeds,
                     no_dictionaries=args.no_dictionaries,
                     oss_fuzz_corpus=args.oss_fuzz_corpus,
                     allow_uncommitted_changes=args.allow_uncommitted_changes,
                     concurrent_builds=concurrent_builds,
                     measurers_cpus=measurers_cpus,
                     runners_cpus=runners_cpus,
                     region_coverage=args.region_coverage,
                     custom_seed_corpus_dir=args.custom_seed_corpus_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
