import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



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

df = pd.read_csv("comb-data.csv")
df["coverage_inc"] = df["edges_covered"] - df["initial_coverage"]

def plot(key_a, key_b):
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
          ]

for ka,kb in toplot:
    plot(ka, kb)
