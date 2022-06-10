import os
import subprocess as sp
import random
import shutil
import glob

trials_per_fuzzertarget = 4
fuzzers = ["afl", "libfuzzer", "aflfast"]
targets = ["bloaty_fuzz_target", "harfbuzz-1.3.2", "lcms-2017-03-21"]
corpus_dirs = ["cminned/bloaty_fuzz_target/aflfast", "cminned/harfbuzz-1.3.2", "cminned/lcms-2017-03-21"]
distribution = "UNIFORM"
seed_usage = 2

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


if len(targets) > 1:
    print()
    print("".join(['*']*80))
    print("".join(['*']*80))
    print("WARNING: running fuzzbench with multiple targets and initial seeds not well \nsupported because it depends on Fuzzbench's ordering of trials")
    print("".join(['*']*80))
    print("".join(['*']*80))
    print()

# Trials are ordered by fuzzer (alphabetic or cli order?), then by target (cli order), then by repetition
# e.g. AFL ,bloaty,0
#      AFL ,bloaty,1
#      AFL ,  lcms,0
#      AFL ,  lcms,1
#      libf,bloaty,0
#      libf,bloaty,1
# etc...


trial_num = 0
for fuzzer in fuzzers:
    for ti, target in enumerate(targets):
        num_seeds = len(corpora_paths[ti])
        seeds_per_trial = int(min(num_seeds / trials_per_fuzzertarget * seed_usage, num_seeds/2))
        seeds_per_trial = max(1, seeds_per_trial) # at least one seed per trial

        # for testing
        # seeds_per_trial = 2

        print()
        print(f"Target is {target}. Fuzzer is {fuzzer}")
        print(f"Approx. {seeds_per_trial} seeds per trial from {distribution} dist")
        print(f"{trials_per_fuzzertarget} unique trials, {total_trials} total trials")
        print("---------------------------------------")

        for tri in range(trials_per_fuzzertarget):
            trial_num = trial_num + 1
            src_trial_num = (trial_num - 1) % (trials_per_fuzzertarget*len(targets)) + 1
            trial_dir = f"seeds/{target}/{trial_num}"

            if trial_num != src_trial_num:
                src_trial_dir = f"seeds/{target}/{src_trial_num}"
                shutil.copytree(src_trial_dir, trial_dir)
                numcpd = len(os.listdir(trial_dir))
                print(f"copied {numcpd} from trial {src_trial_num} to {trial_num}")
                continue
            sp.run(f"mkdir -p {trial_dir}", shell=True)

            if distribution == "UNIFORM":
                trial_num_seeds = random.randint(0, seeds_per_trial*2) # inclusive []
            else:
                trial_num_seeds = seeds_per_trial


            trial_seeds = random.choices(corpus_paths, k=trial_num_seeds)
            for ts in trial_seeds:
                shutil.copy2(ts, trial_dir)
            print(f"sampled {len(os.listdir(trial_dir))} for trial {trial_num}")



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




