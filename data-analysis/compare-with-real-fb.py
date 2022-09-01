import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

fuzzers = ["afl", "libfuzzer", "aflplusplus", "entropic", "honggfuzz"]


def row_rank(row):
    return pd.Series([sum(i < row) + 1 for i in row], name="rank")

def get_mean_ranks(df):
    d = {}
    for f in fuzzers:
        d[f] = df[df["fuzzer"] == f]["edges_covered"]
        d[f] = d[f].reset_index(drop=True)
    d = pd.DataFrame.from_dict(d)

    d_ranks = d.apply(row_rank, axis=1)
    d_ranks.columns = d.columns

    d_mean_ranks = d_ranks.mean()
    d_mean_ranks.index.name = "fuzzer"
    return d_mean_ranks

def get_single_ranks(df):
    d = df.groupby("fuzzer").mean()["edges_covered"]
    d_single_ranks = row_rank(d)
    d_single_ranks.index = d.index
    return d_single_ranks

def labeled_row_rank(a):
    rr = row_rank(a["edges_covered"])
    rr.index = a["fuzzer"]
    rr.name="rank"
    return rr

if not os.path.exists("tmp.csv"):
    df = pd.read_csv("comb-data.csv")
    q = df.groupby(["benchmark", "corpus_size", "mean_size_bytes"])
    ranked = q.apply(labeled_row_rank).reset_index()

    sampled_df = pd.merge(df, ranked[["benchmark", "corpus_size", "mean_size_bytes", "fuzzer", "rank"]], \
            how="left", on=["benchmark", "corpus_size", "mean_size_bytes", "fuzzer"])

    df = pd.read_csv("fb-data.csv")
    df = df[ [f in fuzzers for f in df["fuzzer"]] ]

    srank = df.groupby("benchmark").apply(get_single_ranks)
    mrank = df.groupby("benchmark").apply(get_mean_ranks)

    sampled_df["fbsranks"] = np.NaN
    sampled_df["fbmranks"] = np.NaN
    for i, r in sampled_df.iterrows():
        vs = (srank.loc[r["benchmark"], r["fuzzer"]])
        sampled_df.at[i,'fbsranks'] = vs

        vm = (mrank.loc[r["benchmark"], r["fuzzer"]])
        sampled_df.at[i,'fbmranks'] = vm

    sampled_df.to_csv("tmp.csv")
else:
    sampled_df = pd.read_csv("tmp.csv")

sampled_df["fbsdiff"] = abs(sampled_df["rank"] - sampled_df["fbsranks"])
sampled_df["fbmdiff"] = abs(sampled_df["rank"] - sampled_df["fbmranks"])
m = sampled_df.groupby(["benchmark", "corpus_size", "mean_size_bytes"]).mean().reset_index()

# Rescaled if desired
m['msb_r'] = m.groupby('benchmark')['mean_size_bytes'].apply(lambda x: (x-x.min())/(x.max()-x.min()))
m['mens_r'] = m.groupby('benchmark')['mean_exec_ns'].apply(lambda x: (x-x.min())/(x.max()-x.min()))
m['minit_r'] = m.groupby('benchmark')['initial_coverage'].apply(lambda x: (x-x.min())/(x.max()-x.min()))


for x in ["mean_size_bytes", "mean_exec_ns", "initial_coverage", "q100_mean_size_bytes", "q100_exec_ns"]:
    g = sns.lmplot(data=m, x=x, y="fbmdiff", col="benchmark", sharex=False)
    g.set(ylim=(0, 2.5), ylabel="Difference in Rank to Fuzzbench")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # plt.show()
    plt.savefig(f'assets/compare-w-fb/{x}.png', bbox_inches="tight", \
                dpi = 100)

    plt.close()
    plt.cla()
    plt.clf()


