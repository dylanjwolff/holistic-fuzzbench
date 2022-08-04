import subprocess as sp
import os
import process_e9_logs as pe9l
import time

sut = "sut"
instrums = os.getenv("INSTRMS").split(',')
trial = os.getenv("TRIAL")

all_elapsed = []
all_sizes = []
for path, _, fnames in os.walk("/opt/corpus"):
    for fname in fnames:
        fpath = os.path.join(path, fname)
        for inst in instrums:
            sp.run(f"./{inst} {fpath} | grep NONCE | sort | uniq >> {inst}_log.txt", shell=True)
        start = time.time_ns()
        sp.run(f"./{sut} {fpath}", shell=True)
        end = time.time_ns()
        elapsed = end - start
        all_elapsed.append(elapsed)
        size = os.path.getsize(fpath)
        all_sizes.append(size)

vals = []
vals.append(("per_target_trial", trial))
vals.append(("benchmark", os.getenv("TARGET")))
for inst in instrums:
    if inst in ["eq", "ineq"]:
        ub = pe9l.count_unexplored_branches(f"{inst}_log.txt")
        vals.append((f"{inst}_unexplored", ub))
    ur = pe9l.count_unique_reached(f"{inst}_log.txt")
    vals.append((f"{inst}_reached", ur))
vals.append(("mean_exec_ns", sum(all_elapsed)/len(all_elapsed)))
vals.append(("mean_size_bytes", sum(all_sizes)/len(all_sizes)))
vals.append(("corpus_size", len(all_sizes)))

headers, values = zip(*vals)
print(",".join(headers))
print(",".join([str(v) for v in values]))

os.makedirs("/opt/out/", exist_ok=True)
with open("/opt/out/out.csv", "w") as outf:
    outf.writelines("\n".join([",".join(headers), ",".join([str(v) for v in values])]))
