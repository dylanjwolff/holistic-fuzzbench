import pandas as pd
import seed_stats
import matplotlib
import matplotlib.pyplot as plt

# Report data hard-coded here for now
data = pd.read_csv("report-data/initial-cov-80/data.csv")
seed_data = seed_stats.get_seed_data("seeds/*")
print("WARNING: initial seeds and data CSV are **BOTH** expected to be on local path for now")
data['time_ended'] =  pd.to_datetime(data['time_ended'])
data['time_started'] =  pd.to_datetime(data['time_started'])

mintrials = data.groupby("trial_id").min("time")
mintrials.rename(columns = {'edges_covered':'initial_coverage'}, inplace = True)
mintrials = mintrials.reset_index()

end_data = data.groupby("trial_id").max()

# Seeds hard-coded here for now
seed_data = seed_stats.get_seed_data("seeds/*")

seed_data['trial'] =  pd.to_numeric(seed_data['trial'])

data = pd.merge(end_data, seed_data, how="outer", left_on="trial_id", right_on="trial")
data = pd.merge(data, mintrials[["trial_id", "initial_coverage"]], how="outer", left_on="trial", right_on="trial_id")
print(data.columns)

# Some quick scatter-plots to look at early data
# data.groupby("fuzzer").plot.scatter(x="number of seeds", y="edges_covered")
# plt.show()

# data.groupby("fuzzer").plot.scatter(x="mean seed size", y="edges_covered")
# plt.show()

data.groupby("fuzzer").plot.scatter(x="initial_coverage", y="edges_covered")
plt.show()

data.to_csv("comb-data.csv")
