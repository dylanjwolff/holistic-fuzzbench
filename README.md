# On the Generality of Fuzzer Benchmarking

This artifact is structured as follows:


`fuzzbench-per-trial-seeds` -- A modified version of Fuzzbench which allows seeds files to be provided for each trial. Also contains some pre- and post-processing scripts for gathering data

`final-data-analysis` -- An R Jupyter notebook and Python scripts used to generate the figures in the paper, along with the cleaned and aggregated data in CSV files

`delay` -- Python scripts used to generate the figures for e0 (injecting delays) in the paper

`raw-data` -- The raw files output by Fuzzbench for our experiments

We expect the final data analysis to be most relevant to reviewers.
The notebook and scripts should be runnable on the CSV files in that directory without modification.
For the Python scripts, dependencies are listed in the `requirements.txt` file.
For the R code, all dependencies are installed in the top cell of the notebook. Note that installing these may take some time.
To run the notebook, you will need R with the IRkernel installed (instructions [here](https://irkernel.github.io/installation/)), as well as Jupyter itself.
`fbr.py`, calculates rankings from public fuzzbench data.
We include the public data, but it needs to be decompressed first (e.g. `gunzip fb-paper-data.csv.gz`) 

E0 corresponds to the delay injection experiment (IRQ1).

E1 corresponds to the AFL and LibFuzzer generated corpora discussed in the motivation of the paper (IRQ2).

E2 corresponds to the larger experiment with all corpus / program properties (IRQ3 / IRQ4).
