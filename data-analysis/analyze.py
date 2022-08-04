import pandas as pd
import seed_stats
import matplotlib
import matplotlib.pyplot as plt

# Report data hard-coded here for now
data = pd.read_csv("aggregate.csv")

# Seed stats
seed_stats_names = {"e1v0-afl": "afl-progstats.csv", "e1v0-libfuzzer": "libfuzzer-progstats.csv"}

# Mappings
mappings = pd.read_csv("mappings.csv")
mappings.columns = ["fuzzbench_trial_id", "benchmark", "fuzzer", "per_target_trial"]


ss_dfs = []
for experiment, fname in seed_stats_names.items():
    ss = pd.read_csv(fname)
    ss = pd.merge(ss, mappings, how="left", left_on=["per_target_trial", "benchmark"], right_on=["per_target_trial", "benchmark"])
    ss["experiment"] = experiment
    ss_dfs.append(ss)
merged_ss = pd.concat(ss_dfs, ignore_index=True)

data['time_ended'] =  pd.to_datetime(data['time_ended'])
data['time_started'] =  pd.to_datetime(data['time_started'])

mintrials = data.groupby(["trial_id", "benchmark", "fuzzer"]).min("time")

mintrials.rename(columns = {'edges_covered':'initial_coverage'}, inplace = True)
mintrials = mintrials.reset_index()

end_data = data.groupby(["trial_id", "benchmark", "fuzzer"]).max().reset_index()

# @TODO
# need to merge in seed stats as well

data = pd.merge(end_data, mintrials[["trial_id", "initial_coverage", "benchmark", "fuzzer"]], how="outer", left_on=["trial_id", "benchmark", "fuzzer"], right_on=["trial_id", "benchmark", "fuzzer"])
data = pd.merge(data, merged_ss, how="left", left_on=["trial_id", "benchmark", "fuzzer", "experiment"], right_on=["fuzzbench_trial_id", "benchmark", "fuzzer", "experiment"])
print(data.columns)

# Some quick scatter-plots to look at early data
# data.groupby("fuzzer").plot.scatter(x="number of seeds", y="edges_covered")
# plt.show()

# data.groupby("fuzzer").plot.scatter(x="mean seed size", y="edges_covered")
# plt.show()

# data.groupby("fuzzer").plot.scatter(x="initial_coverage", y="edges_covered")
# plt.show()

data.to_csv("comb-data.csv")
