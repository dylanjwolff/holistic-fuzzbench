import subprocess as sp
import os

sut = os.getenv("BIN")

if sut is None or sut == "":
    p = sp.run('find . -type f -executable -maxdepth 1 | grep -v "afl-fuzz" | grep -v "entry.py"', shell=True, capture_output=True)
    bins = p.stdout.split()
    if len(bins) > 1:
        raise Exception(f"unable to pick out binary from {bins}") 
    sut = bins[0].decode("utf-8")

# aflpp-cmin doesn't like working on docker volumes
sp.run('cp -r /opt/corpus in-corpus', shell=True)
sp.run(f'$AFL_PATH/afl-cmin.bash -i in-corpus -o cminned -t 30000 -m 50000 -e -- {sut} -', shell=True)
sp.run('mv cminned/* /opt/out', shell=True)

