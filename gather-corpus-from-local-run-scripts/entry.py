import subprocess as sp
import os

sut = os.getenv("BIN")
if sut is None or sut == "":
    p = sp.run('find . -type f -executable -maxdepth 1 | grep -v "afl-fuzz" | grep -v "entry.sh"', shell=True, capture_output=True)
    bins = p.stdout.split()
    if len(bins) > 1:
        raise Exception(f"unable to pick out binary from {bins}") 
    os.setenv("BIN", bins[0])

    sp.run('BIN=$(find . -type f -executable -maxdepth 1 | grep -v "afl-fuzz" | grep -v "entry.sh")', shell=True)
sp.run('AFL/afl-cmin -i /opt/corpus -o /opt/out/cminned -t 30000 -e -- $BIN', shell=True)
sp.run('mv /opt/out/cminned/* /opt/out', shell=True)
sp.run('rmdir /opt/out/cminned', shell=True)

