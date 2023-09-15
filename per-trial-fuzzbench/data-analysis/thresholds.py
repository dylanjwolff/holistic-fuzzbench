import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow
import pandas as pd
import random
import itertools
from collections import defaultdict

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

# all "intermediate" data has been filtered out here
response_variables = [
    "edges_covered",
    "coverage_inc",
]

multi_dim = True
grad = True

def plot_ranks_vs_normalized_values(df, props):
    rank_props = [f"{p}_rank" for p in props]
    norm_props = [f"{p}_norm" for p in props]
    df = df[(rank_props + norm_props + ["fuzzer", "benchmark", "trial_id"])]
    print(df.columns)
    print("b4 ^ after v")
    longer_norm = pd.melt(df, id_vars=["fuzzer", "benchmark", "trial_id"] + rank_props, value_vars=norm_props, var_name="Prop.", value_name="Normalized Concrete Value")
    longer_norm["Prop."] = longer_norm["Prop."].map(lambda x: x.replace("_norm", ""))
    longer_rank = pd.melt(df, id_vars=["fuzzer", "benchmark", "trial_id"], value_vars=rank_props, var_name="Prop.", value_name="Rank")
    longer_rank["Prop."] = longer_rank["Prop."].map(lambda x: x.replace("_rank", ""))
    longer = longer_norm.merge(longer_rank, 
        on=["fuzzer", "benchmark", "trial_id", "Prop."],
    )
    print(longer)
    sns.lmplot(longer, x="Normalized Concrete Value", y="Rank", col="Prop.")
    plt.show()
    return

def concrete_rank_diffs_per_subset(per_b_x, prop, norm_prop, rank_diff):
        concrete_diffs = []
        max_rank = per_b_x[norm_prop].max()

        if prop == "initial_coverage":
            max_prop_per_b = per_b_x["edges_covered"].max()
        else:
            max_prop_per_b = per_b_x[prop].max()

        print(f"Property is {prop, norm_prop}")
        # print(sorted(per_b_x[norm_prop]))
        prop_list = (sorted(per_b_x[norm_prop].unique()))
        # print(prop_list)
        diff_s = [np.abs(prop_list[i] - prop_list[j]) for i in range(len(prop_list)-1) for j in range(len(prop_list)-1)]
        unique_diffs = (sorted(list(set(diff_s))))
        print(f"Unique rank differences observed:\n {unique_diffs}")
        closest = min(unique_diffs, key=lambda z: abs(rank_diff - z))
        print(f"Closest observed diff was {closest}")
        print("---")
        rank_diff = closest

        for prop_rank in sorted(per_b_x[norm_prop]):
            if prop_rank + rank_diff <= max_rank:
                concrete_lo = per_b_x[per_b_x[norm_prop] == prop_rank][prop]
                concrete_hi = per_b_x[per_b_x[norm_prop] == prop_rank + rank_diff][prop]
                if len(concrete_hi > 0):
                    concrete_lo = concrete_lo.mean()
                    concrete_hi = concrete_hi.mean()
                    # Convert to scalar ^^^

                    concrete_dif = np.abs(concrete_hi - concrete_lo)
                    norm_concrete_dif = concrete_dif / max_prop_per_b
                    concrete_diffs = concrete_diffs + [norm_concrete_dif]
        return concrete_diffs

def concrete_rank_diffs(x, corpus_props, prog_props, rank_diff, prog_rank_diff):
    norm_p = [f"{p}_{preproc}" for p in corpus_props]

    aggregate_corpus = []
    props = corpus_props
    for i in range(0, len(props)):
        concrete_diffs = [] 
        prop = props[i]
        norm_prop = norm_p[i]

        for b in x["benchmark"].unique():
            per_b_x = x[x["benchmark"]==b]
            per_b_x = per_b_x.groupby("per_target_trial").max()
            concrete_diffs = concrete_diffs + \
                concrete_rank_diffs_per_subset(per_b_x, prop, norm_prop, rank_diff)
            # normlzd_diffs = (np.array(concrete_diffs)/max_prop_per_b)
        normlzd_diffs = np.array(concrete_diffs)
        print(f"\t{normlzd_diffs.mean()} +/- {normlzd_diffs.std()}")
        for nd in normlzd_diffs:
            aggregate_corpus = aggregate_corpus + [{"Property": prop, "(Normalized) Concrete Difference in Values": nd}]

    df = pd.DataFrame(aggregate_corpus)
    print(df)
    sns.displot(df, x="(Normalized) Concrete Difference in Values", col="Property")
    plt.show()

    exit(11)
    props = prog_props
    norm_p = [f"{p}_{preproc}" for p in prog_props]
    for i in range(0, len(props)):
        concrete_diffs = [] 
        prop = props[i]
        norm_prop = norm_p[i]

        per_b_x = x
        concrete_diffs = concrete_diffs + concrete_rank_diffs_per_subset(per_b_x, prop, norm_prop, prog_rank_diff)
            # normlzd_diffs = (np.array(concrete_diffs)/max_prop_per_b)
        normlzd_diffs = np.array(concrete_diffs)
        print(f"\t{normlzd_diffs.mean()} +/- {normlzd_diffs.std()}")
        sns.histplot(normlzd_diffs)
        plt.show()

 
    exit(0)

def thresh(model, x, y, max_corpus_rank, max_program_rank):
    print(f"max corp is {max_corpus_rank}")
    print(f"max prog is {max_program_rank}")
    # "bin_text_size_rank",

    # "mean_exec_ns_rank",
    # "mean_size_bytes_rank", 
    # "initial_coverage_rank"
    props = [
        "mean_exec_ns_rank",
        "mean_size_bytes_rank", 
    ]

    base_prop = props[0]

    fuzzers =  [
        f"fuzzer[T.aflplusplus]",
        f"fuzzer[T.libfuzzer]",
        f"fuzzer[T.entropic]",
    ]

    fuzzer = f"fuzzer[T.aflplusplus]"
    fuzzer = f"fuzzer[T.entropic]"
    fuzzer = None
    fuzzer = f"fuzzer[T.libfuzzer]"

    og_props = props
    if fuzzer is not None:
        fprops = [f"{fuzzer}:{p}" for p in props]
        props = props + fprops

    is_corp = defaultdict(lambda: False)
    for prop in props:
        for base_p in corpus_properties:
            if base_p in prop:
                is_corp[prop] = True
                break

    all_corp = len(is_corp) == len(props)
    all_prog = len(is_corp) == 0
    assert(all_prog or all_corp)

    for f in fuzzers:
        if not f == fuzzer:
            x = x[~x[f].astype(np.bool_)]

    num_rows = len(x.index)

    for jj in range(1,2):
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
                    assert(len(og_props) == 2)
                for rank in range(1, int(np.round(max_corpus_rank))):
                    for rank_inner in range(1, int(np.round(max_corpus_rank))):
                        row2 = row.copy()
                        modd = False
                        for prop in og_props:
                            for prop_inner in og_props:
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
            
        if all_prog:
            for rank in range(1, int(np.round(max_program_rank))):
                row2 = row.copy()
                for prop in props:
                    if not is_corp[prop]:
                        row2[prop] = rank

                all.append(row2)

        x_synthetic = pd.concat(all)

        y_pred = model.predict(x_synthetic)
        y_diverge = np.abs(y_pred - actual)

        data = x_synthetic
        # data["y_diverge"] = y_diverge
        data["y_pred"] = y_pred
        data["norms"] = normz 


        print(normz)
        # print(data[[base_prop, props[1]]])
        print(data[["y_pred", "norms"]])

        # sns.lineplot(data=data, x=base_prop, y="y_diverge")
        # sns.scatterplot(data=data, x=base_prop, y=props[1], hue="y_pred")
        if multi_dim and grad:
            x_max = int(np.round(max_corpus_rank))
            shp = (x_max,x_max)
            print(shp)
            qarr = np.zeros(shp)
        
            for _, row in data.iterrows():
                qarr[int(row[base_prop]), int(row[props[1]])] = row["y_pred"]
            print(qarr)

            sns.heatmap(data=qarr[1:, 1:])
        elif multi_dim:
            sns.lmplot(data=data, x="norms", y="y_pred")
    plt.show()
    print(x_synthetic[prop])

def compute_model_acc(df, preproc, response_preproc, response, fill=0.5, crossval=True, use_boxcox=False):
    corpus_rank_max =  0
    program_rank_max =  0

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
        if corpus_rank_max < df[f"{x}_rank"].max():
            corpus_rank_max = df[f"{x}_rank"].max()

    for x in response_variables: 
        df = df.with_columns([
            (pl.col(x).rank()).over("benchmark").alias(f"{x}_rank"),
        ])

    for x in program_properties: 
        df = df.with_columns([
            (pl.col(x).rank()).alias(f"{x}_rank"),
        ])
        if program_rank_max < df[f"{x}_rank"].max():
            program_rank_max = df[f"{x}_rank"].max()

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

    plot_ranks_vs_normalized_values(pdf, corpus_properties)
    exit(13)
    concrete_rank_diffs(pdf, corpus_properties, program_properties , 50, 800)


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


        thresh(model, x, y, corpus_rank_max, program_rank_max)












        
        return retval, model

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
    return retval


# Use rank or norm to preprocess data
preproc = "rank"
y_preproc = preproc
y_preproc = "final_ranking"
response = "edges_covered"

r, model = compute_model_acc(df, preproc, y_preproc, response, crossval=False, use_boxcox=False)
all_results = r
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



# w = stats.wilcoxon(all_results["test pred. accuracy"] - all_results["naive pred. accuracy"])
# print(f"Wilcoxon p-val = {w}")
# a = a12(all_results["test pred. accuracy"], all_results["naive pred. accuracy"])
# print(f"VDA = {a}")

# grouped per fuzzer pair
# print(all_results.groupby("pair").std()["test pred. accuracy"])
# print(all_results.groupby("pair").std()["naive pred. accuracy"])


