# Explainable Fuzzer Evaluation Artifact

This artifact is structured as follows:

`fuzzbench-per-trial-seeds` -- A modified version of Fuzzbench which allows seeds files to be provided for each trial. Also contains some pre- and post-processing scripts for gathering data
`final-data-analysis` -- An R Jupyter notebook and Python script used to generate the figures in the paper, along with the cleaned and aggregated data in CSV files
`raw-data` -- The raw CSV files output by Fuzzbench for our experiments
