import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow
import pandas as pd
import itertools

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sklearn
from sklearn import linear_model

from sklearn.preprocessing import power_transform
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

import patsy

##
## Data prep / cleaning
##

df = pl.scan_csv("e2-comb-data.csv")
df = pl.read_csv("e2-comb-data.csv")

df = df.filter(pl.col("fuzzer") != "honggfuzz")

df = df = df.with_columns([
    (pl.col("edges_covered") - pl.col("initial_coverage")).alias("coverage_inc")
])

corpus_properties = [
    "corpus_size",
    "initial_coverage",
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

# all "intermediate" data has been filtered out here
response_variables = [
    "edges_covered",
    "coverage_inc",
]

pairs = itertools.combinations(["afl", "libfuzzer", "aflplusplus", "entropic"], 2)

def compute_model_acc(df, pair, preproc, response_preproc, response, fill=0.5):
    ## filter for pairwise analysis
    df = df.filter((pl.col("fuzzer") == pair[0]) | (pl.col("fuzzer") == pair[1]))

    logistic = False

    ## Normalization
    for x in corpus_properties: 
        df = df.with_columns([
            ((pl.col(x) - pl.col(x).min()) / (pl.col(x).max() - pl.col(x).min())).over("benchmark").alias(f"{x}_norm"),
        ])

    for x in response_variables: 
        df = df.with_columns([
            ((pl.col(x) - pl.col(x).min()) / (pl.col(x).max() - pl.col(x).min())).over("benchmark").alias(f"{x}_norm"),
        ])

    for x in program_properties: 
        df = df.with_columns([
            ((pl.col(x) - pl.col(x).min()) / (pl.col(x).max() - pl.col(x).min())).alias(f"{x}_norm"),
        ])

    ## Rank Transform
    for x in corpus_properties: 
        df = df.with_columns([
            (pl.col(x).rank()).over("benchmark").alias(f"{x}_rank"),
        ])

    for x in response_variables: 
        df = df.with_columns([
            (pl.col(x).rank()).over("benchmark").alias(f"{x}_rank"),
        ])

    for x in program_properties: 
        df = df.with_columns([
            (pl.col(x).rank()).alias(f"{x}_rank"),
        ])

    for x in response_variables: 
        df = df.with_columns([
            (pl.col(x).rank()).over(["benchmark", "per_target_trial"]).alias(f"{x}_final_ranking"),
        ])

    y_preproc = response_preproc
    y_name = f"{response}_{y_preproc}"

    # get column names corresponding to preprocessing method
    mean_resp = {}
    for fuzzer in df["fuzzer"].unique():
        subs =  df.filter(pl.col("fuzzer") == fuzzer)
        mean_resp[fuzzer] = np.round(subs[y_name].mean())

    norm_p = [f"{p}_{preproc}" for p in (corpus_properties + program_properties)]

    pdf = df.to_pandas()

    # fill missing
    to_fill = f"indir_reached_{preproc}"
    pdf = pdf.dropna(subset=filter(lambda p: p != to_fill, norm_p))
    pdf = pdf.fillna(value={to_fill: fill})

    x = pdf[["fuzzer"] + norm_p]
    y = pdf[y_name]

    ##
    ## THE MODEL
    ##
    fmla =  f"fuzzer"
    fmla =  f"fuzzer * ( {'+ '.join(norm_p)} )"
    ##
    ##

    x = patsy.dmatrix(fmla, data = x, return_type = "dataframe")


    trn_scores = []
    tst_scores = []
    tst_acc = []
    dumb_acc = []

    # gkf = GroupKFold(n_splits=11) # very high variance when done groupwise
    # for train, test in gkf.split(x, y, groups=pdf["benchmark"]):
    kf = KFold(n_splits=7)
    for train, test in kf.split(x, y):
        print("working on split...")
        x_train, x_test = x.iloc[train], x.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        y_test_orig = y_test
        dumb_preds = np.zeros(len(y_test))
        for fuzzer in mean_resp.keys():
            cname = f"fuzzer[T.{fuzzer}]"
            if cname in x_test.columns:
                dumb_preds = dumb_preds + (x_test[cname]*mean_resp[fuzzer])
            else:
                base = fuzzer
        dumb_preds = dumb_preds + ((dumb_preds == 0)*mean_resp[base])

        if logistic:
            y_train = y_train < 1.5
            y_test = y_test < 1.5

            lgr = linear_model.LogisticRegression()
            model = lgr
        else:
            y_train, lam = stats.boxcox(y_train + 1)
            y_test = stats.boxcox(y_test + 1, lam)

            lrm = linear_model.LinearRegression()
            rfc = RandomForestRegressor()
            gbr = GradientBoostingRegressor()
            mlpr = MLPRegressor()

            model = gbr

        model.fit(x_train, y_train)


        trn_scores = trn_scores + [model.score(x_train, y_train)]
        tst_scores = tst_scores + [model.score(x_test, y_test)]

        if logistic:
            tst_acc = tst_acc + [(y_test == model.predict(x_test)).sum() / len(y_test)]
            dumb_acc = dumb_acc + [(y_test == (dumb_preds < 1.5)).sum() / len(y_test)]
        else:
            tst_acc = tst_acc + [(np.abs(y_test - model.predict(x_test)) <= 0.5).sum() / len(y_test)]
            dumb_acc = dumb_acc + [(np.abs(y_test_orig - dumb_preds) <= 0.5).sum() / len(y_test)]
   
    print(f"Training scores {np.mean(trn_scores)} +/- {np.std(trn_scores)}")
    print(f"Test scores {np.mean(tst_scores)} +/- {np.std(tst_scores)}")
    if "ranking" in y_name:
        print(f"Test acc {np.mean(tst_acc)} +/- {np.std(tst_acc)}")
        print(f"Dumb acc {np.mean(dumb_acc)} +/- {np.std(dumb_acc)}")


    retval = pd.DataFrame([trn_scores, tst_scores, tst_acc, dumb_acc]).T
    retval.columns = ["train score", "test score", "test pred. accuracy", "naive pred. accuracy"]
    retval["pair"] = [pair]*len(trn_scores)
    return retval


# Use rank or norm to preprocess data
preproc = "rank"
y_preproc = preproc
y_preproc = "final_ranking"
response = "coverage_inc"

all_results = []
for pair in pairs:
    r = compute_model_acc(df, pair, preproc, y_preproc, response)
    all_results = [r] + all_results


all_results = pd.concat(all_results)

print(all_results)
print(all_results.mean())
print(all_results.std())
