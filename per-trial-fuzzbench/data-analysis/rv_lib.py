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
from scipy.special import inv_boxcox

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

def compute_model_acc(df, pair, preproc, response_preproc, response, fill=0.5, crossval=True, use_boxcox=False):
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

    dof = []
    trn_scores = []
    adj_scores = []
    tst_scores = []
    tst_acc = []
    dumb_acc = []

    if not crossval:
        y_train = y
        if use_boxcox:
            _, lam = stats.boxcox(y + 1)
            y_train = stats.boxcox(y + 1, lam)
        x_train = x
        model = linear_model.LinearRegression()

        model.fit(x_train, y_train)
        trn_scores = trn_scores + [model.score(x_train, y_train)]
        adj_score = 1 - ( 1-model.score(x_train, y_train) ) * ( len(y_train) - 1 ) / ( len(y_train) - x_train.shape[1] - 1 )
        adj_scores = adj_scores + [adj_score]
        dof = dof + [x_train.shape[0] - x_train.shape[1]]

        retval = pd.DataFrame([trn_scores, adj_scores, dof]).T
        retval.columns = ["train score", "adj score", "dof"]
        retval["pair"] = [pair]*len(trn_scores)
        return retval

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
            if use_boxcox:
                y_train, lam = stats.boxcox(y_train + 1)
                y_test = stats.boxcox(y_test + 1, lam)

            lrm = linear_model.LinearRegression()
            rfc = RandomForestRegressor()
            gbr = GradientBoostingRegressor()
            mlpr = MLPRegressor()

            model = lrm

        model.fit(x_train, y_train)


        trn_scores = trn_scores + [model.score(x_train, y_train)]
        tst_scores = tst_scores + [model.score(x_test, y_test)]

        if logistic:
            tst_acc = tst_acc + [(y_test == model.predict(x_test)).sum() / len(y_test)]
            dumb_acc = dumb_acc + [(y_test == (dumb_preds < 1.5)).sum() / len(y_test)]
        else:
            if use_boxcox:
                tst_acc = tst_acc + [(np.abs(inv_boxcox(y_test, lam) - inv_boxcox(model.predict(x_test), lam)) <= 0.5).sum() / len(y_test)]
                dumb_acc = dumb_acc + [(np.abs(y_test_orig - dumb_preds) <= 0.5).sum() / len(y_test)]
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
response = "edges_covered"

all_results = []
for pair in pairs:
    r = compute_model_acc(df, pair, preproc, y_preproc, response, crossval=False)
    all_results = [r] + all_results
all_results = pd.concat(all_results)
print(all_results)
print(all_results.mean(numeric_only=True))
print(all_results.std(numeric_only=True))

pairs = itertools.combinations(["afl", "libfuzzer", "aflplusplus", "entropic"], 2)
all_results = []
for pair in pairs:
    r = compute_model_acc(df, pair, preproc, y_preproc, response)
    all_results = [r] + all_results


all_results = pd.concat(all_results)

all_results.to_csv("all_res_jun_13.csv")
print(all_results)
print(all_results.mean(numeric_only=True))
print(all_results.std(numeric_only=True))


def a12(measurements_x, measurements_y):
    """Returns Vargha-Delaney A12 measure effect size for two distributions.

    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language effect size statistics
    of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000

    The Vargha and Delaney A12 statistic is a non-parametric effect size
    measure.

    Given observations of a metric (edges_covered or bugs_covered) for
    fuzzer 1 (F2) and fuzzer 2 (F2), the A12 measures the probability that
    running F1 will yield a higher metric than running F2.

    Significant levels from original paper:
      Large   is > 0.714
      Mediumm is > 0.638
      Small   is > 0.556
    """

    x_array = np.asarray(measurements_x)
    y_array = np.asarray(measurements_y)
    x_size, y_size = x_array.size, y_array.size
    ranked = stats.rankdata(np.concatenate((x_array, y_array)))
    rank_x = ranked[0:x_size]  # get the x-ranks

    rank_x_sum = rank_x.sum()
    # A = (R1/n1 - (n1+1)/2)/n2 # formula (14) in Vargha and Delaney, 2000
    # The formula to compute A has been transformed to minimize accuracy errors.
    # See:
    # http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/

    a12_measure = (2 * rank_x_sum - x_size * (x_size + 1)) / (
        2 * y_size * x_size)  # equivalent formula to avoid accuracy errors
    return a12_measure



w = stats.wilcoxon(all_results["test pred. accuracy"] - all_results["naive pred. accuracy"])
print(f"Wilcoxon p-val = {w}")
a = a12(all_results["test pred. accuracy"], all_results["naive pred. accuracy"])
print(f"VDA = {a}")

# grouped per fuzzer pair
# print(all_results.groupby("pair").std()["test pred. accuracy"])
# print(all_results.groupby("pair").std()["naive pred. accuracy"])


