import polars as pl

import sklearn
from sklearn import linear_model
import patsy
import numpy as np
import random
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

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
coefs = pl.read_csv("bootstrap_coeffs.csv")
print(df.columns)
print(coefs)

model = linear_model.LinearRegression()
model.coefs = coefs["estimate"]

# ==================

props = [
    "initial_coverage",
    "mean_exec_ns", 
]

base_prop = props[0]

fuzzers =  [
    f"afl",
    f"aflplusplus",
    f"entropic",
]

fuzzer = f"afl",


df = df.to_pandas()
for fuzzer in fuzzers:
    df[f"fuzzer{fuzzer}"] = (df["fuzzer"] == fuzzer)
    for prop in (corpus_properties + program_properties):
        df[f"fuzzer{fuzzer}:{prop}"] = ((df["fuzzer"] == fuzzer) * df[prop])

column_order = (list(coefs["coefficient"])[1:])
print(column_order)
y = df["fuzzer_rank"]
x = df[column_order]

model.fit(x, y)

model.coefs = coefs["estimate"]

ypred = model.predict(x.iloc[[1]])
print(np.dot(x.to_numpy()[1] , coefs["estimate"].to_numpy()[1:]) + coefs["estimate"][0])
print(ypred)

# ===============

max_corpus_rank = x["mean_exec_ns"].max()
all_corp = True
num_rows = len(x.index)
multi_dim = True
grad = True

fprops = [f"fuzzer{fuzzer}:{p}" for p in props]


is_corp = defaultdict(lambda: False)
for prop in props:
    for base_p in corpus_properties:
        if base_p in prop:
            is_corp[prop] = True
            break
print(num_rows)
sampled_row_num = random.randint(0, num_rows)
row = (x.iloc[[sampled_row_num]])
actual = y.iloc[sampled_row_num]

all = []
normz = []
if all_corp:
    if not multi_dim:
        for rank in range(1, int(np.round(max_corpus_rank))):
            row2 = row.copy()
            for prop in props:
                if is_corp[prop]:
                    row2[prop] = rank
            all.append(row2)
    if multi_dim:
        if grad:
            assert(len(props) == 2)
        for rank in range(1, int(np.round(max_corpus_rank))):
            for rank_inner in range(1, int(np.round(max_corpus_rank))):
                row2 = row.copy()
                modd = False
                for prop in props:
                    for prop_inner in props:
                        if is_corp[prop] and is_corp[prop_inner] and not prop == prop_inner:
                            row2[prop] = rank
                            row2[prop_inner] = rank_inner
                            for fprop in fprops:
                                if prop in fprop:
                                    row2[fprop] = rank
                                if prop_inner in fprop:
                                    row2[fprop] = rank_inner
                            modd = True
                if modd:
                    all.append(row2)
                    normz.append(rank_inner + rank)
            
x_synthetic = pd.concat(all)

y_pred = model.predict(x_synthetic)

data = x_synthetic[column_order]
# data["y_diverge"] = y_diverge
data["y_pred"] = y_pred
data["norms"] = normz 

if multi_dim and grad:
    x_max = int(np.round(max_corpus_rank))
    shp = (x_max,x_max)
    print(shp)
    qarr = np.zeros(shp)

    for _, row in data.iterrows():
        qarr[int(row[base_prop]), int(row[props[1]])] = row["y_pred"]

    sns.heatmap(data=qarr[1:, 1:])
    plt.show()
