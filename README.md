# Seeded Per Trial Fuzzbench

Mounts a different set of seeds as a volume in each container for a local run of Fuzzbench. I expect this will not work in the cloud environment without some tweaking.

Seeds are expeced to be in {{seeds\_per\_trial\_dir}}/{{benchmark}}/{{trial\_id}} on the host machine.

The trials ids start at 1 (not 0) incrementing upwards and are persistent **across experiments**. Thus you must delete the experiment filestore (from the YAML experiment configuration file) before running a "seed per trial" experiment if you have any old experiments that used that location -- even if the old experiments have different names. In theory, multiple targets are supported, but stick to single targets for now until that has been better tested.

Two new variables have been added to the experiment YAML file, exemplified in `example-seed-per-trial-experiment.yaml`.
seeds\_per\_trial\_dir: The location of the seeds on the host system. These should be pre-split by benchmark into trials
use\_seeds\_per\_trial: Feature flag for using a unique initial corpus for each trial.

The feature flag overwrites the `SEED_CORPUS_DIR` environment variable to point to the location of the per-trial seeds rather than Fuzzbench's default intial corpora.
## Overview
<kbd>
  
![FuzzBench Service diagram](docs/images/FuzzBench-service.png)
  
</kbd>


## Sample Report

You can view our sample report
[here](https://www.fuzzbench.com/reports/sample/index.html) and
our periodically generated reports
[here](https://www.fuzzbench.com/reports/index.html).
The sample report is generated using 10 fuzzers against 24 real-world
benchmarks, with 20 trials each and over a duration of 24 hours.
The raw data in compressed CSV format can be found at the end of the report.

When analyzing reports, we recommend:
* Checking the strengths and weaknesses of a fuzzer against various benchmarks.
* Looking at aggregate results to understand the overall significance of the
  result.

Please provide feedback on any inaccuracies and potential improvements (such as
integration changes, new benchmarks, etc.) by opening a GitHub issue
[here](https://github.com/google/fuzzbench/issues/new).

## Documentation

Read our [detailed documentation](https://google.github.io/fuzzbench/) to learn
how to use FuzzBench.

## Contacts

Join our [mailing list](https://groups.google.com/forum/#!forum/fuzzbench-users)
for discussions and announcements, or send us a private email at
[fuzzbench@google.com](mailto:fuzzbench@google.com).
