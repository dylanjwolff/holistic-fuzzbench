import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow
import pandas as pd

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

## filter for pairwise analysis
df = df.filter((pl.col("fuzzer") == "afl") | (pl.col("fuzzer") == "libfuzzer"))

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

##
## Visualizing feature data
##

x = "mean_exec_ns"
b = "zlib_zlib_uncompress_fuzzer"
b = "bloaty_fuzz_target"


# p = df.filter(pl.col("benchmark") == b)[x].unique()
# print(p)

# x = f"{x}_norm"
# eqr = (df[["benchmark", x]].to_pandas())
# sns.displot(eqr, x=x, col="benchmark", col_wrap=5, facet_kws={"sharex":False})
# plt.show()

# x = f"bin_text_size_norm"
# eqr = (df[["benchmark", x]].to_pandas())
# sns.displot(eqr, x=x)
# plt.show()


pdf = df.to_pandas()
# sns.lmplot(data=pdf, x="ineq_unexplored_norm", y="edges_covered_norm", col="benchmark", hue="fuzzer", col_wrap=5)
# plt.show()

# sns.lmplot(data=pdf, x="corpus_size_norm", y="initial_coverage_norm", col="benchmark", col_wrap=5)
# plt.show()

##
## Regression Model
##

# Use rank or norm to preprocess data
preproc = "rank"
y_preproc = preproc

# Response variable can be coverage, coverage increase, or ranking among fuzzers
# w.r.t those metrics 
y_preproc = "final_ranking"
y_name = f"coverage_inc_{y_preproc}"
y_name = f"edges_covered_{y_preproc}"

# need to fill some missing data, value doesn't seem to impact model much at all
fill = .5

# get column names corresponding to preprocessing method
norm_p = [f"{p}_{preproc}" for p in (corpus_properties + program_properties)]

# fill missing
to_fill = f"indir_reached_{preproc}"
pdf = pdf.dropna(subset=filter(lambda p: p != to_fill, norm_p))
pdf = pdf.fillna(value={to_fill: fill})

x = pdf[["fuzzer"] + norm_p]
y = pdf[y_name]

##
## THE MODEL
##
fmla =  f"fuzzer * ( {'+ '.join(norm_p)} )"
fmla =  f"fuzzer"
##
##

x = patsy.dmatrix(fmla, data = x, return_type = "dataframe")

trn_scores = []
tst_scores = []
tst_acc = []
# gkf = GroupKFold(n_splits=11) # very high variance when done groupwise
# for train, test in gkf.split(x, y, groups=pdf["benchmark"]):
kf = KFold(n_splits=7)
for train, test in kf.split(x, y):
    print("working on split...")
    x_train, x_test = x.iloc[train], x.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

    # numerical
    # y_train, lam = stats.boxcox(y_train + 1)
    # y_test = stats.boxcox(y_test + 1, lam)

    # logistic
    y_train = y_train < 1.5
    y_test = y_test < 1.5

    lrm = linear_model.LinearRegression()
    lgr = linear_model.LogisticRegression()
    rfc = RandomForestRegressor()
    gbr = GradientBoostingRegressor()
    mlpr = MLPRegressor()
    model = lgr
    model.fit(x_train, y_train)

    trn_scores = trn_scores + [model.score(x_train, y_train)]
    tst_scores = tst_scores + [model.score(x_test, y_test)]

    # numerical
    # tst_acc = tst_acc + [(np.abs(y_test - model.predict(x_test)) <= 0.5).sum() / len(y_test)]
    # logistic
    tst_acc = tst_acc + [(y_test == model.predict(x_test)).sum() / len(y_test)]

   
print(f"Training scores {np.mean(trn_scores)} +/- {np.std(trn_scores)}")
print(f"Test scores {np.mean(tst_scores)} +/- {np.std(tst_scores)}")
if "ranking" in y_name:
    print(f"Test acc {np.mean(tst_acc)} +/- {np.std(tst_acc)}")


##
## Multicollinearity / Model parsimony checks
##

x = sm.add_constant(x)

def compute_vif(X):
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

omit = [
    "eq_reached",
    "corpus_size",
    "ineq_reached",
    "indir_reached",
    "total_eq",
    "total_indir",
    "total_ineq",
    "q50_mean_size_bytes",
    "q50_exec_ns",
    "mean_size_bytes",
    "mean_exec_ns",

    "q75_mean_size_bytes",
    "q75_exec_ns",

    "ineq_unexplored",
    "eq_unexplored",
]

omit = [f"{o}_{preproc}" for o in omit]

x = x[filter(lambda c: c not in omit, x.columns)]

# print(compute_vif(x))

# y = stats.boxcox(y.to_numpy())
# y = y[0]

model = sm.Logit(y < 1.5, x).fit(method='bfgs')

# model = sm.OLS(y, x).fit()
print(model.summary())


if "const" in x.columns:
    x = x.drop(columns=["const"])
if "Intercept" in x.columns:
    x = x.drop(columns=["Intercept"])
correlations = np.round(x.corr(), decimals=2)

# sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, annot_kws={"size": 6});
# plt.show()

dissimilarity = 1 - abs(correlations)
sf = squareform(dissimilarity)
Z = linkage(sf, 'complete')

dendrogram(Z, labels=x.columns, orientation='top', 
    leaf_rotation=90);
# plt.xticks(fontsize=6, rotation = 45)
# plt.show()

