import os
import subprocess as sp

# for e.g. prev. generated run
# corpus_root = "dest"

# for e.g. oss-fuzz
corpus_root = "oss-corpora"
TARGET_ONLY = True

targets = os.listdir(corpus_root)
# targets = ["bloaty_fuzz_target", "lcms-2017-03-21"]

# Want to throw an error here b/c minimization is v. expensive
os.mkdir("cminned")

ps = []
for target in targets:

    entry = f"""
    BIN=$(find . -type f -executable -maxdepth 1 | grep -v "afl-fuzz" | grep -v "entry.sh")
    AFL/afl-cmin -i /opt/corpus -o /opt/out/cminned -t 30000 -- $BIN
    mv /opt/out/cminned/* /opt/out
    rmdir /opt/out/cminned
    """
    with open("entry.sh", "w") as f:
        f.write(entry)

    dockerignore = f"""
    {corpus_root}
    cminned
    """
    with open(".dockerignore", "w") as f:
        f.write(dockerignore)


    dockerfile = f"""FROM gcr.io/fuzzbench/runners/afl/{target}
        RUN apt update -y
        RUN apt install git -y
        COPY AFL AFL
        RUN cd AFL; make
        ENV AFL_PATH=$WORKDIR/AFL
        COPY entry.sh entry.sh
        RUN chmod a+x entry.sh
        ENTRYPOINT $WORKDIR/entry.sh
    """
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)

    sp.run(f"docker build . -t {target}-cmin", shell=True)

    if TARGET_ONLY:
        cmin_dir = f"{os.getcwd()}/cminned/{target}"
        sp.run(f"mkdir -p {cmin_dir}", shell=True)
        corpus_dir = f"{os.getcwd()}/{corpus_root}/{target}"
        p = sp.Popen(f"docker run -v {corpus_dir}:/opt/corpus -v {cmin_dir}:/opt/out {target}-cmin", shell=True)
        ps.append(p)
        continue

    fuzzers = os.listdir(f"{corpus_root}/{target}")
    # fuzzers = ["libfuzzer"]

    for fuzzer in fuzzers:
        cmin_dir = f"{os.getcwd()}/cminned/{target}/{fuzzer}"
        sp.run(f"mkdir -p {cmin_dir}", shell=True)

        corpus_dir = f"{os.getcwd()}/{corpus_root}/{target}/{fuzzer}"

        sp.run(f"docker run -v {corpus_dir}:/opt/corpus -v {cmin_dir}:/opt/out {target}-cmin", shell=True)

if TARGET_ONLY:
    for p in ps:
        p.wait()
