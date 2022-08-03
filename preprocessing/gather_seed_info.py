import os
import subprocess as sp
import pandas as pd
import glob

# for e.g. prev. generated run
# corpus_root = "dest"

# for e.g. oss-fuzz
corpus_root = "seeds"
TARGET_ONLY = False

targets = os.listdir(corpus_root)
targets = ["bloaty_fuzz_target"]

# Want to throw an error here b/c minimization is v. expensive
os.mkdir("progstats")

ps = []
for target in targets:

    dockerignore = """
    """
    with open(".dockerignore", "w") as f:
        f.write(dockerignore)

        dockerfile = f"""FROM ubuntu:18.04 AS builder
        RUN apt update -y
        RUN apt install xxd -y

        FROM gcr.io/fuzzbench/runners/aflplusplus/{target}
        RUN apt update -y
        RUN apt install git wget unzip markdown -y
        RUN git clone https://github.com/GJDuck/e9patch.git
        COPY --from=builder /usr/bin/xxd /usr/bin/xxd
        RUN cd e9patch; ./build.sh; ./install.sh; dpkg -i e9patch_1.0.0-rc3_amd64.deb
        COPY patches patches
        COPY process_e9_logs.py .
        COPY entry.py entry.py
        ENV TARGET={target}
        COPY get_bin.py get_bin.py
        COPY e9instrum.py e9instrum.py
        RUN python3 get_bin.py
        ENV INSTRMS=eq,ineq,indir,shared
        RUN python3 e9instrum.py
        RUN echo $BIN
        ENTRYPOINT python3 $WORKDIR/entry.py
    """
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)

    sp.run(f"docker build . -t {target}-stats", shell=True, check=True)

    if TARGET_ONLY:
        progstats_dir = f"{os.getcwd()}/progstats/{target}"
        sp.run(f"mkdir -p {progstats_dir}", shell=True)
        corpus_dir = f"{os.getcwd()}/{corpus_root}/{target}"
        p = sp.Popen(f"docker run -v {corpus_dir}:/opt/corpus -v {progstats_dir}:/opt/out {target}-stats", shell=True)
        ps.append(p)
        continue

    for trial in os.listdir(f"{os.getcwd()}/{corpus_root}/{target}/base"):
        progstats_dir = f"{os.getcwd()}/progstats/{target}/{trial}"
        sp.run(f"mkdir -p {progstats_dir}", shell=True)
        corpus_dir = f"{os.getcwd()}/{corpus_root}/{target}/base/{trial}"
        p = sp.Popen(f"docker run -e TRIAL={trial} -v {corpus_dir}:/opt/corpus -v {progstats_dir}:/opt/out {target}-stats", shell=True)
        ps.append(p)


for p in ps:
    p.wait()

sp.run("sudo chown -R $USER progstats", shell=True)
csvs = glob.glob("progstats/**/*.csv", recursive=True)
df = pd.concat((pd.read_csv(f) for f in csvs), ignore_index=True)
print(df)
df.to_csv("progstats.csv")

    
