import os
import subprocess as sp

# for e.g. prev. generated run
# corpus_root = "dest"

# for e.g. oss-fuzz
corpus_root = "oss-corpora"
TARGET_ONLY = True

targets = os.listdir(corpus_root)
targets = ["bloaty_fuzz_target"]

bin_names = {"openssl_x509": "./x509",
             "mbedtls_fuzz_dtlsclient": "./fuzz_dtlsclient",
             "libpcap_fuzz_both": "./fuzz_both"}

# Want to throw an error here b/c minimization is v. expensive
os.mkdir("cminned")

ps = []
for target in targets:

    dockerignore = f"""
    {corpus_root}
    cminned
    """
    with open(".dockerignore", "w") as f:
        f.write(dockerignore)


    if target in bin_names.keys():
        bin_name_str = f"ENV BIN={bin_names[target]}"
    else:
        bin_name_str = ""

    dockerfile = f"""FROM gcr.io/fuzzbench/runners/aflplusplus/{target}
        RUN apt update -y
        RUN apt install git -y
        RUN git clone https://github.com/AFLplusplus/AFLplusplus.git
        RUN cd AFLplusplus; make afl-showmap
        ENV AFL_PATH=$WORKDIR/AFLplusplus
        COPY AFL AFL
        RUN cd AFL; make
        # ENV AFL_PATH=$WORKDIR/AFL
        COPY entry.py entry.py
        RUN chmod a+x entry.py
        {bin_name_str}
        ENV AFL_MAP_SIZE=131072
        ENTRYPOINT python3 $WORKDIR/entry.py
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
