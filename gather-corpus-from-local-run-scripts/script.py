import os
import subprocess as sp

dirs = []
for r, d, f in os.walk("dest"):
    for fname in f:
        old = os.path.join(r, fname)
        new = os.path.join("dest", "_".join((r.replace("/", "_"), fname)).replace("dest_", "")) 
                
        os.rename(old, new)
    for dirname in d:
        dirs.append(os.path.join(r, dirname))

dirs = sorted(dirs, key=len)
dirs.reverse()
for d in dirs:
    os.rmdir(d)
