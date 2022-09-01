# Explainable Fuzzer Evaluation Artifact

This artifact is structured as follows:


`fuzzbench-per-trial-seeds` -- A modified version of Fuzzbench which allows seeds files to be provided for each trial. Also contains some pre- and post-processing scripts for gathering data

`final-data-analysis` -- An R Jupyter notebook and Python script used to generate the figures in the paper, along with the cleaned and aggregated data in CSV files

`raw-data` -- The raw files output by Fuzzbench for our experiments

We expect the final data analysis to be most relevant to reviewers.
The notebook and script should be runnable on the CSV files in that directory without modification.


E1 corresponds to the AFL and LibFuzzer generated corpora discussed in the motivation of the paper.

E2 corresponds to the larger experiment with all corpus / program properties.
