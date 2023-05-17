import glob
import os
import pandas as pd

def get_seed_data(seed_dir):
    data = []
    bench_paths = glob.glob(seed_dir)
    benches = [os.path.basename(p) for p in bench_paths]
    for bench_path in bench_paths:
        trial_paths = glob.glob(os.path.join(bench_path, "*"))
        for trial_path in trial_paths:
            trial = os.path.basename(trial_path)

            seed_paths = glob.glob(os.path.join(trial_path, "*"))
            seed_sizes = [os.path.getsize(sp) for sp in seed_paths]

            mean_seed_size = sum(seed_sizes) / len(seed_sizes)
            num_seeds = len(seed_sizes)
            data.append([trial, mean_seed_size, num_seeds])
    df = pd.DataFrame(data, columns=["trial", "mean seed size", "number of seeds"])
    return df
        # print(f"{trial_path}: {mean_seed_size} {num_seeds}")
        # i.e. we probably want to gather the stats in the runner container -- this should be easy, with the fuzz target in an environment variable? (not sure about how edge coverage is calculated, that actually might be more difficult if it's done separately, but also if we just start recording from 0 this is taken care of)
        # coverage, exec time, (lineage has to be manually annotated)

if __name__ == "__main__":
    print(get_seed_data("seeds/*"))
