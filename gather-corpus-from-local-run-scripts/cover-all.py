import os
import subprocess as sp
import shutil
from tqdm import tqdm

TARGET = "bloaty_fuzz_target"
NPROC = 1

os.mkdir("coverages")

for p in range(NPROC):
    sp.run(f"mkdir -p {TARGET}-coverage-{p}", shell=True)
    sp.run(f"tar -zxvf coverage-binaries/coverage-build-{TARGET}.tar.gz --directory {TARGET}-coverage-{p}", shell=True)


p = 0
for fname in tqdm(os.listdir("dest"))
    sp.run(f"cd {TARGET}-coverage-{p}; ./fuzz_target ../dest/{fname}", shell=True, capture_output=True)
    os.rename(f"{TARGET}-coverage-{p}/default.profraw", f"coverages/{fname}.profraw")
    sp.run(f"llvm-profdata merge coverages/{fname}.profraw --output coverages/{fname}.profdata", shell=True, capture_output=True)