import pandas as pd
import seed_stats
import matplotlib
import matplotlib.pyplot as plt

# Report data hard-coded here for now
data = pd.read_csv("aggregate.csv")

# Seed stats
# seed_stats_names = {"e1v0-afl": "afl-progstats.csv", "e1v0-libfuzzer": "libfuzzer-progstats.csv"}
seed_stats_names = {"oss-sample": "progstats.csv"}

# Mappings
mappings = pd.read_csv("mappings.csv")
mappings.columns = ["fuzzbench_trial_id", "benchmark", "fuzzer", "per_target_trial"]

SINGLE_CORPUS = len(seed_stats_names.keys()) == 1
END_TIME = None
SATURATION = 0.95

ss_dfs = []
for corpus, fname in seed_stats_names.items():
    ss = pd.read_csv(fname)
    ss = pd.merge(mappings, ss, how="left", left_on=["per_target_trial", "benchmark"], right_on=["per_target_trial", "benchmark"])
    ss["corpus"] = corpus
    ss_dfs.append(ss)
merged_ss = pd.concat(ss_dfs, ignore_index=True)
# print(set(merged_ss["fuzzbench_trial_id"]))

data['time_ended'] =  pd.to_datetime(data['time_ended'])
data['time_started'] =  pd.to_datetime(data['time_started'])

mintrials = data.groupby(["trial_id", "benchmark", "fuzzer"]).min("time")

mintrials.rename(columns = {'edges_covered':'initial_coverage'}, inplace = True)
mintrials = mintrials.reset_index()







end_data_bf = data.groupby(["benchmark", "fuzzer"]).max().reset_index()

nd = pd.merge(data, end_data_bf[["benchmark", "fuzzer", "edges_covered"]], how="left", left_on=["benchmark", "fuzzer"], right_on=["benchmark", "fuzzer"], suffixes= ("", "_max"))
nd["saturated"] = nd["edges_covered"] >= nd["edges_covered_max"]*SATURATION

min_sat = nd[nd["saturated"]] \
        .groupby(["trial_id", "benchmark", "fuzzer"])[["time"]] \
        .min().reset_index()

nd = pd.merge(nd, min_sat[["benchmark", "fuzzer", "trial_id", "time"]], how="left", left_on=["trial_id", "benchmark", "fuzzer"], right_on=["trial_id", "benchmark", "fuzzer"], suffixes= ("", "_sat"))
# print(nd[nd["saturated"]][["time_sat", "saturated", "edges_covered", "edges_covered_max"]])


if END_TIME is not None:
    end_data = nd.groupby(["trial_id", "benchmark", "fuzzer"]) \
        .apply(lambda x: x[x["time"] == END_TIME]).reset_index(drop=True)
else:
    end_data = nd.groupby(["trial_id", "benchmark", "fuzzer"]).max().reset_index()



data = pd.merge(end_data, mintrials[["trial_id", "initial_coverage", "benchmark", "fuzzer"]], how="left", left_on=["trial_id", "benchmark", "fuzzer"], right_on=["trial_id", "benchmark", "fuzzer"])

if not SINGLE_CORPUS:
    data["corpus"] = data["experiment"].map(lambda e: "e1v0-afl" if "afl" in e else "e1v0-libfuzzer")
else:
    data["corpus"] = list(seed_stats_names.keys())[0]

data = pd.merge(data, merged_ss, how="left", left_on=["trial_id", "benchmark", "fuzzer", "corpus"], right_on=["fuzzbench_trial_id", "benchmark", "fuzzer", "corpus"])
print(data)

# Some quick scatter-plots to look at early data
# data.groupby("fuzzer").plot.scatter(x="number of seeds", y="edges_covered")
# plt.show()

# data.groupby("fuzzer").plot.scatter(x="mean seed size", y="edges_covered")
# plt.show()

# data.groupby("fuzzer").plot.scatter(x="initial_coverage", y="edges_covered")
# plt.show()

if END_TIME is not None:
    data.to_csv(f"{END_TIME}_comb-data.csv")
else:
    data.to_csv("comb-data.csv")
