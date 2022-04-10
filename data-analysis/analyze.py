import pandas as pd
import seed_stats
import matplotlib
import matplotlib.pyplot as plt

# Report data hard-coded here for now
data = pd.read_csv("reports/spt-80/data.csv")
data['time_ended'] =  pd.to_datetime(data['time_ended'])
data['time_started'] =  pd.to_datetime(data['time_started'])

end_data = data.groupby("trial_id").max()

# Seeds hard-coded here for now
seed_data = seed_stats.get_seed_data("seeds/*")

seed_data['trial'] =  pd.to_numeric(seed_data['trial'])

data = pd.merge(end_data, seed_data, how="outer", left_on="trial_id", right_on="trial")
print(data.columns)


# Some quick scatter-plots to look at early data
data.groupby("fuzzer").plot.scatter(x="number of seeds", y="edges_covered")
# plt.show()

data.groupby("fuzzer").plot.scatter(x="mean seed size", y="edges_covered")
# plt.show()

data.to_csv("comb-data.csv")
