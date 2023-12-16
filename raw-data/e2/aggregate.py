import polars as pl
import glob
import subprocess as sp
import shutil
import os

corpus_properties = [
    "corpus_size",
    "eq_reached",
    "eq_unexplored",
    "ineq_reached",
    "ineq_unexplored",
    "indir_reached",
    "mean_exec_ns",
    "q25_exec_ns",
    "q50_exec_ns",
    "q75_exec_ns",
    "q100_exec_ns",
    "mean_size_bytes",
    "q25_mean_size_bytes",
    "q50_mean_size_bytes",
    "q75_mean_size_bytes",
    "q100_mean_size_bytes",
]
# omitting because of many missing values:
# "shared_reached",

program_properties = [
    "total_shared",
    "total_eq",
    "total_ineq",
    "total_indir",
    "bin_text_size",
]

other = [
    "per_target_trial"
]

pk = ["fuzzer", "benchmark", "trial_id"]


def read_from_archives():
    archive_fnames = glob.glob("*.tar.gz")
    dfs = []
    for fname in archive_fnames:
        dirname = fname.replace(".tar.gz", "_out")

        os.makedirs(f"{dirname}")
        sp.run(f"cd {dirname}; tar -xvf ../{fname}", shell=True)
        inner_dirname = f"{dirname}/{os.listdir(dirname)[0]}"

        csv_gz_fname = (glob.glob(f"{inner_dirname}/**/*.csv.gz")[0])
        sp.run(f"gunzip {csv_gz_fname}", shell=True)
        dfs += [pl.read_csv(f'{csv_gz_fname.replace(".gz", "")}')]
        shutil.rmtree(f"{dirname}")

    df = pl.concat(dfs)
    return df

# df = read_from_archives()
# df.write_csv("raw_aggregate.csv")
df = pl.read_csv("raw_aggregate.csv")
df = df.filter(pl.col("fuzzer") != "honggfuzz")
init = df.filter(pl.col("time") == 0) \
    .rename({"edges_covered": "initial_coverage"}) \
    .select(pk + ["initial_coverage"])

df = df.filter(pl.col("time") == 86400)
df = df.join(init, on=pk)

seed_stats = pl.read_csv("e2-comb-data.csv")
seed_stats = seed_stats.select(pk + corpus_properties + program_properties + other)

df = df.join(seed_stats, on=pk)
df.write_csv("aggregate.csv")







