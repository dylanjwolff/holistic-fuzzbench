import pandas as pd
import seed_stats

data = pd.read_csv("reports/spt-80/data.csv")
data['time_ended'] =  pd.to_datetime(data['time_ended'])
data['time_started'] =  pd.to_datetime(data['time_started'])
print(data.columns)

end_data = data.groupby("trial_id").max("time")
seed_data = seed_stats.get_seed_data("seeds/*")

seed_data['trial'] =  pd.to_numeric(seed_data['trial'])

data = pd.merge(end_data, seed_data, left_on="trial_id", right_on="trial")
print(data)
