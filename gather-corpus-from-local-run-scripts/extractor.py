import os
import subprocess as sp

files = os.listdir("oss-corpora")
zipfiles = [f"oss-corpora/{f}" for f in files if ".zip" in f]
bases = [f.split(".zip")[0] for f in zipfiles]

for base in bases:
    os.makedirs(base, exist_ok=True)

for zipfile, base in zip(zipfiles, bases):
    sp.run(f"mv {zipfile} {base}", shell=True)

bases = [f"oss-corpora/{f}" for f in os.listdir("oss-corpora")]
print(bases)
for base in bases:
    sp.run(f"cd {base}; unzip *.zip; rm *.zip", shell=True)
    sp.run(f"rdfind -deleteduplicates true {base}", shell=True)

