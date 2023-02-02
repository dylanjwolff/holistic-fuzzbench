# Holistic Fuzzer Evaluation Artifact

This artifact is structured as follows:


`fuzzbench-per-trial-seeds` -- A modified version of Fuzzbench which allows seeds files to be provided for each trial. Also contains some pre- and post-processing scripts for gathering data

`final-data-analysis` -- An R Jupyter notebook and Python scripts used to generate the figures in the paper, along with the cleaned and aggregated data in CSV files

`raw-data` -- The raw files output by Fuzzbench for our experiments

We expect the final data analysis to be most relevant to reviewers.
The notebook and scripts should be runnable on the CSV files in that directory without modification.
For the Python scripts, dependencies are listed in the `requirements.txt` file.
For the R code, all dependencies are installed in the top cell of the notebook. Note that installing these may take some time.
To run the notebook, you will need R with the IRkernel installed (instructions [here](https://irkernel.github.io/installation/)), as well as Jupyter itself.
`fbr.py`, calculates rankings from public fuzzbench data.
We include the public data, but it needs to be decompressed first (`gunzip fb-paper-data.csv.gz`) 


E1 corresponds to the AFL and LibFuzzer generated corpora discussed in the motivation of the paper (section 2.1).

E2 corresponds to the larger experiment with all corpus / program properties.
