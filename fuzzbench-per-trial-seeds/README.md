# Seeded Per Trial Fuzzbench

Mounts a different set of seeds as a volume in each container for a local run of Fuzzbench. I expect this will not work in the cloud environment without some tweaking.

Seeds are expeced to be in {{seeds\_per\_trial\_dir}}/{{benchmark}}/{{trial\_id}} on the host machine.

Two new variables have been added to the experiment YAML file, exemplified in `example-seed-per-trial-experiment.yaml`.

`seeds\_per\_trial\_dir`: The location of the seeds on the host system. These should be pre-split by benchmark into trials

`use\_seeds\_per\_trial`: Feature flag for using a unique initial corpus for each trial.

The feature flag overwrites the `SEED_CORPUS_DIR` environment variable to point to the location of the per-trial seeds rather than Fuzzbench's default intial corpora.


To run, follow the Fuzzbench documented instructions [here](https://google.github.io/fuzzbench/running-a-local-experiment) and set additional configuration as desired.
