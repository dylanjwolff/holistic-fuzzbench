import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import scipy.stats as ss

df = pl.read_csv("smartslow.csv")
df_2 = pl.read_csv("aflslow.csv")
df = pl.concat([df, df_2])


df = df.select(["fuzzer", "benchmark", "trial_id", "time", "edges_covered"])

df = df.with_columns(
    (pl.col("fuzzer").str.contains("aflpp") |
     pl.col("fuzzer").str.contains("aflplusplus")).alias("AFL++")
)

df = df.with_columns(
    (pl.col("fuzzer").str.contains("afl") & ~pl.col("AFL++")).alias("AFL")
)

df = df.with_columns(
    (pl.col("fuzzer").str.contains("eclipser")).alias("Eclipser")
)

df = df.with_columns(
    (pl.col("fuzzer").str.contains("rsymsan")).alias("RSymSan (Jigsaw)")
)

df = df.melt(id_vars=["fuzzer", "benchmark", "trial_id", "time", "edges_covered"], variable_name = "base fuzzer") \
    .filter(pl.col("value")).drop("value")


df = df.with_columns(
    (~pl.col("fuzzer").str.contains("10")).alias("0")
)

df = df.with_columns(
    pl.col("fuzzer").str.contains("10k").alias("10")
)

df = df.with_columns(
    pl.col("fuzzer").str.contains("100k").alias("100")
)

df = df.with_columns(
    pl.col("fuzzer").str.contains("1000").alias("1")
)

df = df.melt(id_vars=["base fuzzer", "fuzzer", "benchmark", "trial_id", "time", "edges_covered"], variable_name = "slowdown") \
    .filter(pl.col("value")).drop(["value", "fuzzer"])


def plot_slowdowns(df):
    df = df.with_columns(
        pl.col("trial_id").mod(20).alias("trial_group_id")
    )

    slowdowns = df["slowdown"].unique()

    diffed_slowdowns = []
    decrease_colname = "Difference in Edges Covered"
    slowdown_colname = "Slowdown (ms)"
    for slowdown in slowdowns:
        r = df.filter(pl.col("slowdown").is_in(["0", slowdown])) \
            .sort("slowdown") \
            .group_by(["base fuzzer", "benchmark", "time", "trial_group_id"]) \
            .agg([
                pl.col("edges_covered").diff().alias(decrease_colname)
            ]).explode(decrease_colname) \
            .filter(~pl.col(decrease_colname).is_null()) \
            .with_columns(pl.lit(int(slowdown)).alias(slowdown_colname))

        diffed_slowdowns += [r]

    diffed_slowdowns = pl.concat(diffed_slowdowns)
    print(diffed_slowdowns)

    sns.relplot(diffed_slowdowns.to_pandas(),
        x="time",
        y=decrease_colname,
        kind="line",
        hue="base fuzzer",
        col=slowdown_colname)

    plt.show()

# plot_slowdowns(df)
# exit()

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
    ranked = ss.rankdata(np.concatenate((x_array, y_array)))
    rank_x = ranked[0:x_size]  # get the x-ranks

    rank_x_sum = rank_x.sum()
    # A = (R1/n1 - (n1+1)/2)/n2 # formula (14) in Vargha and Delaney, 2000
    # The formula to compute A has been transformed to minimize accuracy errors.
    # See:
    # http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/

    a12_measure = (2 * rank_x_sum - x_size * (x_size + 1)) / (
        2 * y_size * x_size)  # equivalent formula to avoid accuracy errors
    return a12_measure

def calc_paired_stat(df, f, mirror=False, inverse=True):
    fuzzers = df["base fuzzer"].unique()

    combos = itertools.combinations(fuzzers, 2)

    agg = []
    for a, b in combos:

        # u_a = df.filter(pl.col("base fuzzer") == a)["edges_covered"].mean()
        # u_b = df.filter(pl.col("base fuzzer") == b)["edges_covered"].mean()
        # print(a, b)
        # print(u_a, u_b)
        stat = f(
            df.filter(pl.col("base fuzzer") == a)["edges_covered"],
            df.filter(pl.col("base fuzzer") == b)["edges_covered"]
        )
        agg += [(a, b, stat)]
        assert(not (mirror and inverse))
        if mirror:
            agg += [(b, a, stat)]
        if inverse:
            agg += [(b, a, 1 - stat)]

    stat_df = pd.DataFrame(agg, columns = ["f1", "f2", "stat"]).set_index(["f1", "f2"]).T.stack().T #.T.stack().T
    return (stat_df)

def mwu(x, y):
    return ss.mannwhitneyu(x, y).pvalue

def bf_adj(df, alpha):
    h, l = df.shape
    assert(h == l)
    num_tests = h * l / 2 - h
    print(f"BF adj. p-value is {alpha/num_tests}")
    p_adj = df < (alpha/num_tests)
    p = df < alpha
    # if assertion fails, lost statistical power
    #    can try bonferoni-holmes or other more powerful methods rather
    #    than vanilla bf adjustment
    assert((p == p_adj).all().all())
    return p_adj

df = df.filter(pl.col("time") == df["time"].max()).drop("time")
base_df = df.filter(pl.col("slowdown") == "0").drop("slowdown")
slowest_df = df.filter(pl.col("slowdown") == "100").drop("slowdown")

print(calc_paired_stat(base_df, a12))
print(
    bf_adj(
        calc_paired_stat(base_df, mwu, mirror=True, inverse=False),
        0.05
    ))
print(calc_paired_stat(base_df, mwu, mirror=True, inverse=False))
print(calc_paired_stat(slowest_df, a12))
print(
    bf_adj(
        calc_paired_stat(slowest_df, mwu, mirror=True, inverse=False),
        0.05
    ))
print(calc_paired_stat(slowest_df, mwu, mirror=True, inverse=False))


