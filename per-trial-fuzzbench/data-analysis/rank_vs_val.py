import polars as pl

import sklearn
from sklearn import linear_model
import patsy
import numpy as np
import random
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from collections import defaultdict

corpus_properties = [
    # "corpus_size",
    "initial_coverage",
    # "eq_reached",
    # "eq_unexplored",
    # "ineq_reached",
    # "ineq_unexplored",
    # "indir_reached",
    "mean_exec_ns",
    # "q25_exec_ns",
    # "q50_exec_ns",
    # "q75_exec_ns",
    # "q100_exec_ns",
    "mean_size_bytes",
    # "q25_mean_size_bytes",
    # "q50_mean_size_bytes",
    # "q75_mean_size_bytes",
    # "q100_mean_size_bytes",
]
# omitting because of many missing values:
# "shared_reached",

program_properties = [
    # "total_shared",
    # "total_eq",
    # "total_ineq",
    # "total_indir",
    "bin_text_size",
]

# =================

df = pl.read_csv("fitted_data.csv", infer_schema_length=1500)
raw = pl.read_csv("e2-comb-data.csv", infer_schema_length=1500)
raw = raw.select(["per_target_trial", "benchmark"] + corpus_properties + program_properties)
raw = raw.groupby(["per_target_trial", "benchmark"]).mean()

normalized = raw
for x in corpus_properties: 
    normalized = normalized.with_columns([
        ((pl.col(x) - pl.col(x).min()) / (pl.col(x).max() - pl.col(x).min()))
            .over("benchmark").alias(f"{x}_norm"),
    ])
for x in program_properties: 
    normalized = normalized.with_columns([
        ((pl.col(x) - pl.col(x).min()) / (pl.col(x).max() - pl.col(x).min()))
            .alias(f"{x}_norm"),
    ])

normalized = normalized.with_columns((
    pl.col("per_target_trial").cast(pl.Int64)
))

df = normalized.join(df, on=["per_target_trial", "benchmark"], suffix="_rank")
# df = df.select(
#     [f"{p}_norm" for p in corpus_properties + program_properties]
#     + [f"{p}_rank" for p in corpus_properties + program_properties]
# )
cols = (
    [f"{p}_norm" for p in corpus_properties + program_properties]
    + [f"{p}_rank" for p in corpus_properties + program_properties]
)
df = df.melt(id_vars = ["per_target_trial", "benchmark"], value_vars=cols)

df = df.with_columns(
    pl.col("variable").apply(lambda s: "norm" if "_norm" in s else "rank").alias("class")
)
df = df.with_columns(
    pl.col("variable").apply(lambda s: s.replace("_norm", "").replace("_rank", ""))
)

df = (df.pivot(values="value", index=["per_target_trial", "benchmark", "variable"], columns="class"))

sns.lmplot(df.to_pandas(), x = "norm", y = "rank", row="variable")
plt.show()



    
