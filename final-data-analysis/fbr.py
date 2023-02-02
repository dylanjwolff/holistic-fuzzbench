import polars as pl
from scipy.stats import mannwhitneyu
from itertools import combinations

df = pl.read_csv("fb-paper-data.csv")
df = df.join(df.select(["time", "benchmark"]).groupby("benchmark").max(), on="benchmark", suffix="_max")
mean_edges = df.filter(pl.col("time") >= pl.col("time_max")).select(["edges_covered", "benchmark", "fuzzer"]).groupby(["benchmark", "fuzzer"]).mean()
mean_edges = mean_edges.to_pandas()
mean_edges = mean_edges[mean_edges["fuzzer"].isin(["afl", "libfuzzer", "aflplusplus", "entropic"])]


benchs_from_ours = [ "freetype2-2017", "harfbuzz-1.3.2", "libjpeg-turbo-07-2017", "libpcap_fuzz_both", "libxslt_xpath", "libpng-1.2.56", "sqlite3_ossfuzz", "vorbis-2017-12-11", "woff2-2016-05-06", "zlib_zlib_uncompress_fuzzer", "mbedtls_fuzz_dtlsclient"]

mean_edges = mean_edges[mean_edges["benchmark"].isin(benchs_from_ours)]

print(mean_edges)
ranks = mean_edges \
    .join(mean_edges.groupby(["benchmark"]).rank(), rsuffix="_rank")
print(ranks.groupby("fuzzer").mean())


print()
for combo in combinations(["afl", "libfuzzer", "aflplusplus", "entropic"], 2):
    x = ranks[ranks["fuzzer"] == combo[0]]["edges_covered_rank"]
    y = ranks[ranks["fuzzer"] == combo[1]]["edges_covered_rank"]
    r = mannwhitneyu(x, y)
    print(combo)
    print(f"    {r.pvalue}")
print()
