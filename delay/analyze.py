import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

df = pl.read_csv("data.csv")
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

df = df.with_columns(
    pl.col("trial_id").mod(20).alias("trial_group_id")
    
)

slowdowns = df["slowdown"].unique()

diffed_slowdowns = []
decrease_colname = "Difference in Edges Covered"
slowdown_colname = "Slowdown (microseconds)"
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