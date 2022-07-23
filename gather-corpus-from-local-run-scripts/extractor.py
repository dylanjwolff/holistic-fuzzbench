import os
import subprocess as sp

files = os.listdir()
zipfiles = [f for f in files if ".zip" in f]
bases = [f.split(".zip")[0] for f in zipfiles]
for base in bases:
    os.makedirs(base, exist_ok=True)

for zipfile, base in zip(zipfiles, bases):
    sp.run(f"mv {zipfile} {base}", shell=True)

bases = os.listdir()
print(bases)
for base in bases:
    sp.run(f"cd {base}; unzip *.zip; rm *.zip", shell=True)

