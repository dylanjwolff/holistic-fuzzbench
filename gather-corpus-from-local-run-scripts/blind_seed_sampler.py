import os
import subprocess as sp
import random
import shutil
import glob
import numpy as np

trials_per_fuzzertarget = 20
fuzzers = ["afl", "libfuzzer", "aflplusplus", "entropic", "honggfuzz"]
targets = ["bloaty_fuzz_target", "harfbuzz-1.3.2", "lcms-2017-03-21"]
corpus_dirs = ["cminned/bloaty_fuzz_target/aflfast", "cminned/harfbuzz-1.3.2", "cminned/lcms-2017-03-21"]

seed_root = "aflpp-edge-cminned-oss"
targets = os.listdir(seed_root)
corpus_dirs = [f"{seed_root}/{target}" for target in targets]

# distribution = "UNIFORM"
distribution = "EXP"
mean_seed_usage = 0.2

assert(len(corpus_dirs) == len(targets))
sp.run("rm -r seeds", shell=True)

total_trials = trials_per_fuzzertarget*len(fuzzers)*len(targets)

corpora_paths = []
for corpus_dir in corpus_dirs:
    recf = [f for f in glob.glob(f"{corpus_dir}/**/*") if os.path.isfile(f)]
    dfs = [f for f in glob.glob(f"{corpus_dir}/*") if os.path.isfile(f)]
    corpus_paths = list(set(recf + dfs))
    corpora_paths.append(corpus_paths)
    print(f"Sampling from {len(corpus_paths)} files under {corpus_dir}")

for ti, target in enumerate(targets):
    num_seeds = len(corpora_paths[ti])

    seeds_per_trial = num_seeds * mean_seed_usage
    seeds_per_trial = max(1, seeds_per_trial)  # at least one seed per trial
    seeds_per_trial = min(num_seeds, seeds_per_trial)  # no more than exists

    # for testing
    # seeds_per_trial = 20

    print(f"\nTarget is {target}")
    print(f"Approx. {int(seeds_per_trial)} seeds per trial of {num_seeds} total, from {distribution} dist")
    print(f"{trials_per_fuzzertarget} unique trials, {total_trials} total trials")
    print("---------------------------------------")

    for tri in range(trials_per_fuzzertarget):
        trial_dir = f"seeds/{target}/base/{tri}"
        sp.run(f"mkdir -p {trial_dir}", shell=True)
        if distribution == "UNIFORM":
            trial_num_seeds = random.randint(1, int(np.round(seeds_per_trial*2))) # inclusive []
        elif distribution == "EXP":
            trial_num_seeds = int(np.round(np.random.exponential(scale=seeds_per_trial, size=1)[0]))
        else:
            trial_num_seeds = seeds_per_trial
        trial_num_seeds = min(trial_num_seeds, num_seeds) # no more than exists

        trial_seeds = random.choices(corpus_paths, k=trial_num_seeds)
        for ts in trial_seeds:
            shutil.copy2(ts, trial_dir)

        print(f"sampled {len(os.listdir(trial_dir))} for fb trial {tri}")


    for fuzzer in fuzzers:
        shutil.copytree(f"seeds/{target}/base", f"seeds/{target}/{fuzzer}")



# for target in targets:
#     for i in range(1, trials_per_fuzzertarget+1):
#         if distribution == "UNIFORM":
#             trial_num_seeds = random.randint(0, seeds_per_trial*2) # inclusive []
#         else:
#             trial_num_seeds = seeds_per_trial
# 
#         trial_dir = f"seeds/{target}/{i}"
#         sp.run(f"mkdir -p {trial_dir}", shell=True)
# 
#         trial_seeds = random.choices(corpus_paths, k = trial_num_seeds)
#         for ts in trial_seeds:
#             shutil.copy2(ts, trial_dir)
#
#     for j in range(trials_per_fuzzertarget+1, total_trials+1):
#         trial_dir = f"seeds/{target}/{j}"
#         src_trial_dir = f"seeds/{target}/{((j-1)%trials_per_fuzzertarget)+1}"
#         shutil.copytree(src_trial_dir, trial_dir)




