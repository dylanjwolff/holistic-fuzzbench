import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

e2 = pl.read_csv("comb-data.csv")
fb = pl.read_csv("fb-data.csv")
print(fb["corpus_size"])

var = "corpus_size"

df = e2.select(["benchmark", var]).unique().join(fb, on = "benchmark").select(["benchmark", var, f"{var}_right"]).to_pandas()

axs = df.hist(column=var, by="benchmark")

for r in axs:
    for ax in r:
        l = df[df["benchmark"] == ax.title._text][f"{var}_right"].unique()[0]
        ax.axvline(l, color="r", linestyle="-")


plt.show()


