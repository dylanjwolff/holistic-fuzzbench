import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
from pandas.api.types import is_numeric_dtype

sns.set(font_scale=1.5)

def spear(df, a_key, b_key, p = False, c = True, unstack = True):
    if p and c:
        f = lambda rs: ss.spearmanr(rs[a_key], rs[b_key])
    elif p:
        f = lambda rs: ss.spearmanr(rs[a_key], rs[b_key]).pvalue
    elif c:
        f = lambda rs: ss.spearmanr(rs[a_key], rs[b_key]).correlation
    else:
        raise Exception("must show p-value or correlation")

    o = df.groupby(["fuzzer", "benchmark"]) \
            .apply(f)

    if unstack:
        return o.unstack(level=-1)
    else:
        return o

df = pd.read_csv("e2-comb-data.csv")
df["coverage_inc"] = df["edges_covered"] - df["initial_coverage"]
# df = pd.read_csv("e2-group-scaled.csv")

predictors = ["initial_coverage", "mean_size_bytes", "mean_exec_ns", "ineq_unexplored", "eq_unexplored", "indir_reached", "corpus_size",  "q100_mean_size_bytes",  "q100_exec_ns", "q25_mean_size_bytes", "q50_mean_size_bytes", "q75_mean_size_bytes", "q25_exec_ns", "q50_exec_ns", "q75_exec_ns"]


remap_preds = {
        "mean_size_bytes":  "Mean Seed Size",
        "mean_exec_ns": "Mean Exec Time",
        "initial_coverage": "Initial Coverage",
        "bin_text_size": "Program Size",
        "corpus_size": "Corpus Size",
 }

for p in predictors:
    if p == "bin_text_size":
        continue
    df[p] = df.groupby('benchmark')[p].apply(lambda x: (x-x.min())/(x.max()-x.min()))
df["edges_covered"] = df.groupby(['benchmark', 'fuzzer'])["edges_covered"].apply(lambda x: (x-x.min())/(x.max()-x.min()))

def plot(df, key_a, key_b):
    o = (spear(df, key_a, key_b))
    o = o.round(decimals=2)
    op = (spear(df, key_a, key_b, p=True, c=False))
    psig = 0.05

    print(op<psig)
    ax = sns.heatmap(o[op<psig], annot=True, vmin=-1, vmax=1)
    ax = sns.heatmap(o[op>psig], annot=True, cbar=False, 
                        cmap=sns.color_palette("Greys", n_colors=1, desat=1))

    colors = [sns.color_palette("Greys", n_colors=1, desat=1)[0]]
    texts = [f"p>{psig})"]
    patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.10, -.15), loc='center')

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)


    ax.set_xlabel("Benchmarks")
    ax.set_ylabel("Fuzzers")
    ax.set_title(f"Spearman's Rho for {key_a} and {key_b}")

    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    # plt.show()
    plt.savefig(f'assets/spearman-heatmaps/{key_a}_{key_b}.png', bbox_inches="tight", \
        dpi = 100)

    plt.close()
    plt.cla()
    plt.clf()

toplot = [("initial_coverage", "coverage_inc"),\
          ("mean_size_bytes", "coverage_inc"),\
          ("mean_exec_ns", "coverage_inc"),\
          ("ineq_unexplored", "initial_coverage"),\
          ("eq_unexplored", "initial_coverage"),\
          ("indir_reached", "initial_coverage"),\
          ("corpus_size", "initial_coverage"),\
          ("mean_exec_ns", "initial_coverage"),\
          ("mean_exec_ns", "edges_covered"),\
          ("mean_size_bytes", "initial_coverage"),\
          ("mean_size_bytes", "edges_covered"),\
          ("q25_mean_size_bytes", "coverage_inc"),\
          ("q25_mean_size_bytes", "initial_coverage"),\
          ("q50_mean_size_bytes", "coverage_inc"),\
          ("q75_mean_size_bytes", "coverage_inc"),\
          ("q100_mean_size_bytes", "coverage_inc"),\
          ("q100_mean_size_bytes", "q100_exec_ns"),\
          ("q100_mean_size_bytes", "initial_coverage"),\
          ("q100_exec_ns", "initial_coverage"),\
          ("q25_exec_ns", "coverage_inc"),\
          ("q25_exec_ns", "initial_coverage"),\
          ("q50_exec_ns", "coverage_inc"),\
          ("q75_exec_ns", "coverage_inc"),\
          ("q100_exec_ns", "coverage_inc"),\
          ("q100_exec_ns", "initial_coverage"),\
          ]

# for ka,kb in toplot:
#    plot(df, ka, kb)



o = df[list(remap_preds.keys()) + ["edges_covered"]].corr(method="spearman")
o = o.round(decimals=2)

o = o[:-1]
c = o["edges_covered"]
o["Edges Covered"] = c
o["edges_covered"] = np.NaN
o = o.rename(columns={"edges_covered": ""})
o = o.rename(index=remap_preds, columns=remap_preds)

m = o*0
m[""] = np.ones(len(m[""]))
m = m > 0


def remap(labels):
    [remap_preds[l] for l in labels]

ax = sns.heatmap(o, annot=True, vmin=-1, vmax=1, center=0, cmap="RdBu", mask=m)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
ax.xaxis.set_ticks_position('none') 


figure = plt.gcf() # get current figure
figure.set_size_inches(10, 6)

plt.savefig(f'assets/spearman-heatmaps/all-pred.png', bbox_inches="tight", \
    dpi = 100)
plt.show()
