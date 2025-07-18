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
coefs = pl.read_csv("bootstrap_coeffs.csv")
print(df.columns)
print(coefs)

model_coefs = coefs["estimate"]
model = linear_model.LinearRegression()
model.coefs = model_coefs

# ==================

props = [
    "initial_coverage",
    "mean_exec_ns", 
]

fuzzers =  [
    f"afl",
    f"aflplusplus",
    f"entropic",
]

curr_fuzzer = f"entropic"

fprops = [f"fuzzer{curr_fuzzer}:{p}" for p in props] if curr_fuzzer != "libfuzzer" else []

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

model.coefs = model_coefs 

model_pred = model.predict(x.iloc[[1]])
def predict(x):
    return np.dot(x.to_numpy(), model_coefs.to_numpy()[1:]) + model_coefs[0]
# print(model_pred, predict(x.iloc[[1]]))
# assert(model_pred == predict(x.iloc[[1]]))

# ===============

for f in fuzzers:
    if not f == fuzzer:
        x = x[~x[f"fuzzer{f}"].astype(np.bool_)]

max_corpus_rank = x["mean_exec_ns"].max()
max_program_rank = x["bin_text_size"].max()
all_corp = True
num_rows = len(x.index)
multi_dim = True
grad = True


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


prop_maxes = [int(np.round(
                  max_corpus_rank if is_corp[p] else max_program_rank)) 
             for p in props]

print(is_corp[props[1]])
print(prop_maxes)

all = []
normz = []
if all_corp:
    if not multi_dim:
        for rank in range(1, prop_maxes[0]):
            row2 = row.copy()
            for prop in props:
                if is_corp[prop]:
                    row2[prop] = rank
            all.append(row2)
    if multi_dim:
        if grad:
            assert(len(props) == 2)
        for rank in range(1, prop_maxes[0] + 1):
            for rank_inner in range(1, prop_maxes[1] + 1):
                row2 = row.copy()
                row2[props[0]] = rank
                row2[props[1]] = rank_inner
                print(f"set {props[0]} to {rank} and {props[1]} to {rank_inner}")
                for fprop in fprops:
                    if props[0] in fprop:
                        print(f"set {fprop} to {rank}")
                        row2[fprop] = rank
                    if props[1] in fprop:
                        row2[fprop] = rank_inner
                        print(f"set {fprop} to {rank_inner}")
                print("----------")
                all.append(row2)
                normz.append(rank_inner + rank)
            
x_synthetic = pd.concat(all)

data = x_synthetic[column_order]
print((data.iloc[-1]))
print(predict(data.iloc[-1]))
y_pred = predict(data)

# data["y_diverge"] = y_diverge
data["y_pred"] = y_pred
data["norms"] = normz 

if multi_dim and grad:
    shp = (prop_maxes[0] + 1,prop_maxes[1] + 1)

    print(shp)
    qarr = np.zeros(shp)

    for _, row in data.iterrows():
        qarr[int(row[props[0]]), int(row[props[1]])] = row["y_pred"]

    qarr = qarr[1:, 1:]
    min_val = qarr.min()
    mask = (qarr - min_val >= 0.5)
    # inv_mask = (qarr - min_val < 0.5)

    threshold = min_val + 0.5
    norm_threshold = 0.5 / np.abs(qarr.max() - min_val)

    if threshold < qarr.max():
        colors = ["blue", "white", "purple"]
        nodes = [0, norm_threshold, 1]
        cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

        print(f"min val {min_val}")
        print(f"thresh is {threshold}")
        print(f"max is {qarr.max()}")
        sns.heatmap(data=qarr, cmap=cmap)
        # sns.heatmap(data=qarr, mask=mask)
    else:
        colors = ["blue", "white"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)
        sns.heatmap(data=qarr, cmap=cmap)
    plt.show()
