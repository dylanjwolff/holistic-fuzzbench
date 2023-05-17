import os
import subprocess as sp
import shutil
import random
import json
from tqdm import tqdm

THRESH = 84000
TARGET = "bloaty_fuzz_target"
NPROC = 1
TEMPDIR = "temp"
os.mkdir(TEMPDIR)

cover_bin_dir = f"{TEMPDIR}/{TARGET}-coverage"

sp.run(f"mkdir -p {cover_bin_dir}", shell=True)

def get_branch_cov(f):
    o = sp.run(f"llvm-cov export --instr-profile {f} {cover_bin_dir}/fuzz_target --summary-only", shell=True, capture_output=True, text=True)
    cov = json.loads(o.stdout)
    return cov["data"][0]["totals"]["branches"]["count"]

def compute_coverage(fname):
    sp.run(f"./{cover_bin_dir}/fuzz_target dest/{fname}", shell=True, capture_output=True)
    return "default.profraw"

def merge_coverage(fname, coverage, cum_coverage=""):
    sp.run(f"llvm-profdata merge {fname} {cum_coverage} --output new_cov.profdata", shell=True) # , capture_output=True)
    return "new_cov.profdata"


sp.run(f"tar -zxvf coverage-binaries/coverage-build-{TARGET}.tar.gz --directory {cover_bin_dir}", shell=True)

files = os.listdir("dest")

coverage=0
merged = ""
while coverage < THRESH and len(files) > 0:
    random.shuffle(files)
    f = files.pop()
    covf = compute_coverage(f)
    merged = merge_coverage(covf, merged)
    os.delete(covf)
    coverage = get_branch_cov(merged)
    print(f"total coverage {coverage}")
print(f"Finished with total coverage {coverage}")
shutil.rmtree("temp")