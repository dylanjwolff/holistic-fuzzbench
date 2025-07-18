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
"""Runs fuzzer for trial."""

import importlib
import json
import glob
import os
import posixpath
import random
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile

import numpy

from common import benchmark_config
from common import environment
from common import experiment_utils
from common import filesystem
from common import filestore_utils
from common import fuzzer_utils
from common import fuzzer_stats
from common import logs
from common import new_process
from common import retry
from common import sanitizer
from common import utils

NUM_RETRIES = 3
RETRY_DELAY = 3

FUZZ_TARGET_DIR = os.getenv('OUT', '/out')

CORPUS_ELEMENT_BYTES_LIMIT = 1 * 1024 * 1024
SEED_CORPUS_ARCHIVE_SUFFIX = '_seed_corpus.zip'

fuzzer_errored_out = False  # pylint:disable=invalid-name

CORPUS_DIRNAME = 'corpus'
RESULTS_DIRNAME = 'results'
CORPUS_ARCHIVE_DIRNAME = 'corpus-archives'


def _clean_seed_corpus(seed_corpus_dir):
    """Prepares |seed_corpus_dir| for the trial. This ensures that it can be
    used by AFL which is picky about the seed corpus. Moves seed corpus files
    from sub-directories into the corpus directory root. Also, deletes any files
    that exceed the 1 MB limit. If the NO_SEEDS env var is specified than the
    seed corpus files are deleted."""
    if not os.path.exists(seed_corpus_dir):
        return

    if environment.get('NO_SEEDS'):
        logs.info('NO_SEEDS specified, deleting seed corpus files.')
        shutil.rmtree(seed_corpus_dir)
        os.mkdir(seed_corpus_dir)
        return

    failed_to_move_files = []
    for root, _, files in os.walk(seed_corpus_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            if os.path.getsize(file_path) > CORPUS_ELEMENT_BYTES_LIMIT:
                os.remove(file_path)
                logs.warning('Removed seed file %s as it exceeds 1 Mb limit.',
                             file_path)
                continue

            sha1sum = utils.file_hash(file_path)
            new_file_path = os.path.join(seed_corpus_dir, sha1sum)
            try:
                shutil.move(file_path, new_file_path)
            except OSError:
                failed_to_move_files.append((file_path, new_file_path))

    if failed_to_move_files:
        logs.error('Failed to move seed corpus files: %s', failed_to_move_files)


def get_clusterfuzz_seed_corpus_path(fuzz_target_path):
    """Returns the path of the clusterfuzz seed corpus archive if one exists.
    Otherwise returns None."""
    fuzz_target_without_extension = os.path.splitext(fuzz_target_path)[0]
    seed_corpus_path = (fuzz_target_without_extension +
                        SEED_CORPUS_ARCHIVE_SUFFIX)
    return seed_corpus_path if os.path.exists(seed_corpus_path) else None


def _copy_custom_seed_corpus(corpus_directory):
    """Copy custom seed corpus provided by user"""
    shutil.rmtree(corpus_directory)
    benchmark = environment.get('BENCHMARK')
    benchmark_custom_corpus_dir = posixpath.join(
        experiment_utils.get_custom_seed_corpora_filestore_path(), benchmark)
    filestore_utils.cp(benchmark_custom_corpus_dir,
                       corpus_directory,
                       recursive=True)


def _unpack_clusterfuzz_seed_corpus(fuzz_target_path, corpus_directory):
    """If a clusterfuzz seed corpus archive is available, unpack it into the
    corpus directory if it exists. Copied from unpack_seed_corpus in
    engine_common.py in ClusterFuzz.
    """
    oss_fuzz_corpus = environment.get('OSS_FUZZ_CORPUS')
    if oss_fuzz_corpus:
        benchmark = environment.get('BENCHMARK')
        corpus_archive_filename = f'{benchmark}.zip'
        oss_fuzz_corpus_archive_path = posixpath.join(
            experiment_utils.get_oss_fuzz_corpora_filestore_path(),
            corpus_archive_filename)
        seed_corpus_archive_path = posixpath.join(FUZZ_TARGET_DIR,
                                                  corpus_archive_filename)
        filestore_utils.cp(oss_fuzz_corpus_archive_path,
                           seed_corpus_archive_path)
    else:
        seed_corpus_archive_path = get_clusterfuzz_seed_corpus_path(
            fuzz_target_path)

    if not seed_corpus_archive_path:
        return

    with zipfile.ZipFile(seed_corpus_archive_path) as zip_file:
        # Unpack seed corpus recursively into the root of the main corpus
        # directory.
        idx = 0
        for seed_corpus_file in zip_file.infolist():
            if seed_corpus_file.filename.endswith('/'):
                # Ignore directories.
                continue

            # Allow callers to opt-out of unpacking large files.
            if seed_corpus_file.file_size > CORPUS_ELEMENT_BYTES_LIMIT:
                continue

            output_filename = f'{idx:016d}'
            output_file_path = os.path.join(corpus_directory, output_filename)
            zip_file.extract(seed_corpus_file, output_file_path)
            idx += 1

    logs.info('Unarchived %d files from seed corpus %s.', idx,
              seed_corpus_archive_path)


def sample_corpus(corpus_dir,
                  random_seed,
                  dest_dir=None,
                  distribution='EXP',
                  mean_seed_util=0.2):
    """Samples a pseudo-random number of files from the input corpus. By
    default sampling is done in-place, destructively, removing unsampled
    files. Will sample (mean_seed_util * number of seeds) files on
    average, according to the specified distribution. Sampling is
    deterministic wrt the random seed"""

    gen = random.Random(random_seed)
    npgen = numpy.random.RandomState(random_seed)

    inplace = dest_dir is None

    corpus_paths = [
        f for f in glob.glob(f'{corpus_dir}/**/*', recursive=True)
        if os.path.isfile(f)
    ]
    corpus_paths.sort()  # need to be ordered for deterministic sampling
    logs.info('Sampling from %d files in seed corpus dir %s.',
              len(corpus_paths), corpus_dir)

    num_seeds = len(corpus_paths)

    if distribution == 'UNIFORM':
        trial_num_seeds = gen.randint(
            1, int(numpy.round(mean_seed_util * num_seeds)))  # inclusive []
    elif distribution == 'EXP':
        trial_num_seeds = int(
            numpy.round(
                npgen.exponential(scale=(mean_seed_util * num_seeds),
                                  size=1)[0]))
    else:
        raise Exception('Unimplemented sampling algorithm')

    trial_num_seeds = min(trial_num_seeds, num_seeds)  # no more than exists

    trial_seeds = gen.sample(corpus_paths, k=trial_num_seeds)

    if inplace:
        for path in corpus_paths:
            if path not in trial_seeds:
                os.remove(path)
    else:
        for path in trial_seeds:
            shutil.copy(path, dest_dir)


def run_fuzzer(max_total_time, log_filename):
    """Runs the fuzzer using its script. Logs stdout and stderr of the fuzzer
    script to |log_filename| if provided."""
    input_corpus = environment.get('SEED_CORPUS_DIR')
    output_corpus = os.environ['OUTPUT_CORPUS_DIR']
    fuzz_target_name = environment.get('FUZZ_TARGET')
    target_binary = fuzzer_utils.get_fuzz_target_binary(FUZZ_TARGET_DIR,
                                                        fuzz_target_name)
    if not target_binary:
        logs.error('Fuzz target binary not found.')
        return

    if max_total_time is None:
        logs.warning('max_total_time is None. Fuzzing indefinitely.')

    runner_niceness = environment.get('RUNNER_NICENESS', 0)

    # Set sanitizer options environment variables if this is a bug based
    # benchmark.
    env = None
    benchmark = environment.get('BENCHMARK')
    if benchmark_config.get_config(benchmark).get('type') == 'bug':
        env = os.environ.copy()
        sanitizer.set_sanitizer_options(env, is_fuzz_run=True)

    try:
        # Because the runner is launched at a higher priority,
        # set it back to the default(0) for fuzzing processes.
        command = [
            'nice', '-n',
            str(0 - runner_niceness), 'python3', '-u', '-c',
            (f'from fuzzers.{environment.get("FUZZER")} import fuzzer; '
             'fuzzer.fuzz('
             f'"{shlex.quote(input_corpus)}", "{shlex.quote(output_corpus)}", '
             f'"{shlex.quote(target_binary)}")')
        ]

        # Write output to stdout if user is fuzzing from command line.
        # Otherwise, write output to the log file.
        if environment.get('FUZZ_OUTSIDE_EXPERIMENT'):
            new_process.execute(command,
                                timeout=max_total_time,
                                write_to_stdout=True,
                                kill_children=True,
                                env=env)
        else:
            with open(log_filename, 'wb') as log_file:
                new_process.execute(command,
                                    timeout=max_total_time,
                                    output_file=log_file,
                                    kill_children=True,
                                    env=env)
    except subprocess.CalledProcessError:
        global fuzzer_errored_out  # pylint:disable=invalid-name
        fuzzer_errored_out = True
        logs.error('Fuzz process returned nonzero.')


class TrialRunner:  # pylint: disable=too-many-instance-attributes
    """Class for running a trial."""

    def __init__(self):
        self.fuzzer = environment.get('FUZZER')
        if not environment.get('FUZZ_OUTSIDE_EXPERIMENT'):
            benchmark = environment.get('BENCHMARK')
            trial_id = environment.get('TRIAL_ID')
            self.gcs_sync_dir = experiment_utils.get_trial_bucket_dir(
                self.fuzzer, benchmark, trial_id)
            filestore_utils.rm(self.gcs_sync_dir, force=True, parallel=True)
        else:
            self.gcs_sync_dir = None

        self.cycle = 0
        self.output_corpus = environment.get('OUTPUT_CORPUS_DIR')
        self.corpus_archives_dir = os.path.abspath(CORPUS_ARCHIVE_DIRNAME)
        self.results_dir = os.path.abspath(RESULTS_DIRNAME)
        self.log_file = os.path.join(self.results_dir, 'fuzzer-log.txt')
        self.last_sync_time = None
        self.last_archive_time = -float('inf')

    def initialize_directories(self):
        """Initialize directories needed for the trial."""
        directories = [
            self.output_corpus,
            self.corpus_archives_dir,
            self.results_dir,
        ]

        for directory in directories:
            filesystem.recreate_directory(directory)

    def set_up_corpus_directories(self):
        """Set up corpora for fuzzing. Set up the input corpus for use by the
        fuzzer and set up the output corpus for the first sync so the initial
        seeds can be measured."""
        fuzz_target_name = environment.get('FUZZ_TARGET')
        target_binary = fuzzer_utils.get_fuzz_target_binary(
            FUZZ_TARGET_DIR, fuzz_target_name)
        input_corpus = environment.get('SEED_CORPUS_DIR')
        os.makedirs(input_corpus, exist_ok=True)
        if not environment.get('CUSTOM_SEED_CORPUS_DIR'):
            _unpack_clusterfuzz_seed_corpus(target_binary, input_corpus)
        else:
            _copy_custom_seed_corpus(input_corpus)

        _clean_seed_corpus(input_corpus)

        corpus_variant_id = environment.get('CORPUS_VARIANT_ID')
        seed_sample_distribution = environment.get('SEED_SAMPLE_DIST')
        mean_seed_util = environment.get('SEED_SAMPLE_MEAN_UTIL')
        random_seed = environment.get('RANDOMNESS_SEED')

        mean_seed_util = float(
            mean_seed_util) if mean_seed_util is not None else 0.2
        random_seed = int(random_seed) if random_seed is not None else 0
        corpus_variant_id = int(corpus_variant_id) \
            if corpus_variant_id is not None else 0

        if seed_sample_distribution is not None:
            sample_corpus(input_corpus,
                          random_seed + corpus_variant_id,
                          distribution=seed_sample_distribution,
                          mean_seed_util=mean_seed_util)

        # Ensure seeds are in output corpus.
        os.rmdir(self.output_corpus)
        shutil.copytree(input_corpus, self.output_corpus)

    def conduct_trial(self):
        """Conduct the benchmarking trial."""
        self.initialize_directories()

        logs.info('Starting trial.')

        self.set_up_corpus_directories()

        max_total_time = environment.get('MAX_TOTAL_TIME')
        args = (max_total_time, self.log_file)

        # Sync initial corpus before fuzzing begins.
        self.do_sync()

        fuzz_thread = threading.Thread(target=run_fuzzer, args=args)
        fuzz_thread.start()
        if environment.get('FUZZ_OUTSIDE_EXPERIMENT'):
            # Hack so that the fuzz_thread has some time to fail if something is
            # wrong. Without this we will sleep for a long time before checking
            # if the fuzz thread is alive.
            time.sleep(5)

        while fuzz_thread.is_alive():
            self.cycle += 1
            self.sleep_until_next_sync()
            self.do_sync()

        logs.info('Doing final sync.')
        self.do_sync()
        fuzz_thread.join()

    def sleep_until_next_sync(self):
        """Sleep until it is time to do the next sync."""
        if self.last_sync_time is not None:
            next_sync_time = (self.last_sync_time +
                              experiment_utils.get_snapshot_seconds())
            sleep_time = next_sync_time - time.time()
            if sleep_time < 0:
                # Log error if a sync has taken longer than
                # get_snapshot_seconds() and messed up our time
                # synchronization.
                logs.warning('Sleep time on cycle %d is %d', self.cycle,
                             sleep_time)
                sleep_time = 0
        else:
            sleep_time = experiment_utils.get_snapshot_seconds()
        logs.debug('Sleeping for %d seconds.', sleep_time)
        time.sleep(sleep_time)
        # last_sync_time is recorded before the sync so that each sync happens
        # roughly get_snapshot_seconds() after each other.
        self.last_sync_time = time.time()

    def do_sync(self):
        """Save corpus archives and results to GCS."""
        try:
            self.archive_and_save_corpus()
            # TODO(metzman): Enable stats.
            self.save_results()
            logs.debug('Finished sync.')
        except Exception:  # pylint: disable=broad-except
            logs.error('Failed to sync cycle: %d.', self.cycle)

    def record_stats(self):
        """Use fuzzer.get_stats if it is offered, validate the stats and then
        save them to a file so that they will be synced to the filestore."""
        # TODO(metzman): Make this more resilient so we don't wait forever and
        # so that breakages in stats parsing doesn't break runner.

        fuzzer_module = get_fuzzer_module(self.fuzzer)

        fuzzer_module_get_stats = getattr(fuzzer_module, 'get_stats', None)
        if fuzzer_module_get_stats is None:
            # Stats support is optional.
            return

        try:
            output_corpus = environment.get('OUTPUT_CORPUS_DIR')
            stats_json_str = fuzzer_module_get_stats(output_corpus,
                                                     self.log_file)

        except Exception:  # pylint: disable=broad-except
            logs.error('Call to %s failed.', fuzzer_module_get_stats)
            return

        try:
            fuzzer_stats.validate_fuzzer_stats(stats_json_str)
        except (ValueError, json.decoder.JSONDecodeError):
            logs.error('Stats are invalid.')
            return

        stats_filename = experiment_utils.get_stats_filename(self.cycle)
        stats_path = os.path.join(self.results_dir, stats_filename)
        with open(stats_path, 'w', encoding='utf-8') as stats_file_handle:
            stats_file_handle.write(stats_json_str)

    def archive_corpus(self):
        """Archive this cycle's corpus."""
        archive = os.path.join(
            self.corpus_archives_dir,
            experiment_utils.get_corpus_archive_name(self.cycle))

        with tarfile.open(archive, 'w:gz') as tar:
            new_archive_time = self.last_archive_time
            for file_path in get_corpus_elements(self.output_corpus):
                try:
                    stat_info = os.stat(file_path)
                    last_modified_time = stat_info.st_mtime
                    if last_modified_time <= self.last_archive_time:
                        continue  # We've saved this file already.
                    new_archive_time = max(new_archive_time, last_modified_time)
                    arcname = os.path.relpath(file_path, self.output_corpus)
                    tar.add(file_path, arcname=arcname)
                except (FileNotFoundError, OSError):
                    # We will get these errors if files or directories are being
                    # deleted from |directory| as we archive it. Don't bother
                    # rescanning the directory, new files will be archived in
                    # the next sync.
                    pass
                except Exception:  # pylint: disable=broad-except
                    logs.error('Unexpected exception occurred when archiving.')
        self.last_archive_time = new_archive_time
        return archive

    def save_corpus_archive(self, archive):
        """Save corpus |archive| to GCS and delete when done."""
        if not self.gcs_sync_dir:
            return

        basename = os.path.basename(archive)
        gcs_path = posixpath.join(self.gcs_sync_dir, CORPUS_DIRNAME, basename)

        # Don't use parallel to avoid stability issues.
        filestore_utils.cp(archive, gcs_path)

        # Delete corpus archive so disk doesn't fill up.
        os.remove(archive)

    @retry.wrap(NUM_RETRIES, RETRY_DELAY,
                'experiment.runner.TrialRunner.archive_and_save_corpus')
    def archive_and_save_corpus(self):
        """Archive and save the current corpus to GCS."""
        archive = self.archive_corpus()
        self.save_corpus_archive(archive)

    @retry.wrap(NUM_RETRIES, RETRY_DELAY,
                'experiment.runner.TrialRunner.save_results')
    def save_results(self):
        """Save the results directory to GCS."""
        if not self.gcs_sync_dir:
            return
        # Copy results directory before rsyncing it so that we don't get an
        # exception from uploading a file that changes in size. Files can change
        # in size because the log file containing the fuzzer's output is in this
        # directory and can be written to by the fuzzer at any time.
        results_copy = filesystem.make_dir_copy(self.results_dir)
        filestore_utils.rsync(
            results_copy, posixpath.join(self.gcs_sync_dir, RESULTS_DIRNAME))


def get_fuzzer_module(fuzzer):
    """Returns the fuzzer.py module for |fuzzer|. We made this function so that
    we can mock the module because importing modules makes hard to undo changes
    to the python process."""
    fuzzer_module_name = f'fuzzers.{fuzzer}.fuzzer'
    fuzzer_module = importlib.import_module(fuzzer_module_name)
    return fuzzer_module


def get_corpus_elements(corpus_dir):
    """Returns a list of absolute paths to corpus elements in |corpus_dir|."""
    corpus_dir = os.path.abspath(corpus_dir)
    corpus_elements = []
    for root, _, files in os.walk(corpus_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            corpus_elements.append(file_path)
    return corpus_elements


def experiment_main():
    """Do a trial as part of an experiment."""
    logs.info('Doing trial as part of experiment.')
    try:
        runner = TrialRunner()
        runner.conduct_trial()
    except Exception as error:  # pylint: disable=broad-except
        logs.error('Error doing trial.')
        raise error


def main():
    """Do an experiment on a development machine or on a GCP runner instance."""
    logs.initialize(
        default_extras={
            'benchmark': environment.get('BENCHMARK'),
            'component': 'runner',
            'fuzzer': environment.get('FUZZER'),
            'trial_id': str(environment.get('TRIAL_ID')),
        })
    experiment_main()
    if fuzzer_errored_out:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
