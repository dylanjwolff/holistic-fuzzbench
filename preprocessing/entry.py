import subprocess as sp
import os
import process_e9_logs as pe9l
import time
import numpy as np

sut = "sut"
instrums = os.getenv("INSTRMS").split(',')
trial = os.getenv("TRIAL")

all_elapsed = []
all_sizes = []
N = 30
for path, _, fnames in os.walk("/opt/corpus"):
    for fname in fnames:
        fpath = os.path.join(path, fname)
        size = os.path.getsize(fpath)
        if size > 1000:
            continue

        cum_elapsed = N

        for i in range(N):
            start = time.time_ns()
            sp.run(f"timeout 2s ./{sut} {fpath}", shell=True)
            end = time.time_ns()
            elapsed = end - start
            cum_elapsed = elapsed + cum_elapsed
        # if cum_elapsed / 1e9 > N:
        #    continue

        all_elapsed.append(cum_elapsed / N)
        all_sizes.append(size)

        for inst in instrums:
            sp.run(f"timeout 2s ./{inst} {fpath} | grep NONCE | sort | uniq >> {inst}_log.txt", shell=True)



vals = []
vals.append(("per_target_trial", trial))
vals.append(("benchmark", os.getenv("TARGET")))

ineq_str = "jg\|jnle\|jle\|jng\|jl\|jnge\|jge\|jnl\|jbe\|jna\|ja\|jnbe\|jb\|jnae\|jc\|jnb\|jae\|jnc\|js\|jns"
eq_str = "je\|jne\|jz\|jnz"
js_str = f"{ineq_str}\|{eq_str}"

slc = sp.run(f'objdump -d ./{sut} | grep "@plt" | wc -l', shell=True, capture_output=True)
vals.append(("total_shared", slc.stdout.decode("utf-8").strip()))
eq = sp.run(f'objdump -d ./{sut} | grep "{eq_str}" | wc -l', shell=True, capture_output=True)
vals.append(("total_eq", eq.stdout.decode("utf-8").strip()))
ineq = sp.run(f'objdump -d ./{sut} | grep "{ineq_str}" | wc -l', shell=True, capture_output=True)
vals.append(("total_ineq", ineq.stdout.decode("utf-8").strip()))
indir = sp.run(f'objdump -d ./{sut} | grep "{js_str}\|call" | grep "%r" | wc -l', shell=True, capture_output=True)
vals.append(("total_indir", indir.stdout.decode("utf-8").strip()))
size = sp.run('objdump -h ./{sut} | grep .text | tr -s " " | cut -f 4 -d " "', shell=True, capture_output=True)
vals.append(("bin_text_size", str(int(indir.stdout.decode("utf-8").strip(), base=16))))
print("binsize")
print(str(int(indir.stdout.decode("utf-8").strip(), base=16)))


for inst in instrums:
    if inst in ["eq", "ineq"]:
        ub = pe9l.count_unexplored_branches(f"{inst}_log.txt")
        vals.append((f"{inst}_unexplored", ub))
    ur = pe9l.count_unique_reached(f"{inst}_log.txt")
    vals.append((f"{inst}_reached", ur))

vals.append(("mean_exec_ns", sum(all_elapsed)/len(all_elapsed)))
vals.append(("q25_exec_ns", np.quantile(all_elapsed, 0.25)))
vals.append(("q50_exec_ns", np.quantile(all_elapsed, 0.50)))
vals.append(("q75_exec_ns", np.quantile(all_elapsed, 0.75)))
vals.append(("q100_exec_ns", np.quantile(all_elapsed, 1)))

vals.append(("mean_size_bytes", sum(all_sizes)/len(all_sizes)))
vals.append(("q25_mean_size_bytes", np.quantile(all_sizes, 0.25)))
vals.append(("q50_mean_size_bytes", np.quantile(all_sizes, 0.50)))
vals.append(("q75_mean_size_bytes", np.quantile(all_sizes, 0.75)))
vals.append(("q100_mean_size_bytes", np.quantile(all_sizes, 1)))

vals.append(("corpus_size", len(all_sizes)))

headers, values = zip(*vals)
print(",".join(headers))
print(",".join([str(v) for v in values]))

os.makedirs("/opt/out/", exist_ok=True)
with open("/opt/out/out.csv", "w") as outf:
    outf.writelines("\n".join([",".join(headers), ",".join([str(v) for v in values])]))
