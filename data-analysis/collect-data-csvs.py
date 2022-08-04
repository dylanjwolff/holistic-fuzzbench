import glob
import subprocess as sp
import pandas as pd
import os

datafiles = glob.glob("**/*.csv.gz", recursive=True)

print(datafiles)

dfs = []
for zipped in datafiles:
    csv = zipped.rsplit(".", 1)[0]
    if not os.path.exists(csv):
        sp.run(f"gunzip -k {zipped}", shell=True)
    if not os.path.exists(csv):
        raise Exception(f"file should exist {csv}")
    dfs.append(pd.read_csv(csv))

df = pd.concat(dfs, ignore_index=True)
print(df)
df.to_csv("aggregate.csv")
