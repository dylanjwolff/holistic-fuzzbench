import os
import subprocess as sp

targets = os.listdir("dest")
# targets = ["bloaty_fuzz_target"]
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
    dest
    cminned
    """
    with open(".dockerignore", "w") as f:
        f.write(dockerignore)


    dockerfile = f"""FROM gcr.io/fuzzbench/runners/afl/{target}
        RUN apt install git -y
        RUN git clone https://github.com/google/AFL.git
        RUN cd AFL; make
        ENV AFL_PATH=$WORKDIR/AFL
        COPY entry.sh entry.sh
        RUN chmod a+x entry.sh
        ENTRYPOINT $WORKDIR/entry.sh
    """
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)

    sp.run(f"docker build . -t {target}-cmin", shell=True)

    fuzzers = os.listdir(f"dest/{target}")
    # fuzzers = ["libfuzzer"]
    for fuzzer in fuzzers:
        cmin_dir = f"{os.getcwd()}/cminned/{target}/{fuzzer}"
        sp.run(f"mkdir -p {cmin_dir}", shell=True)

        corpus_dir = f"{os.getcwd()}/dest/{target}/{fuzzer}"

        sp.run(f"docker run -v {corpus_dir}:/opt/corpus -v {cmin_dir}:/opt/out {target}-cmin", shell=True)
