import os
import subprocess as sp
import shutil
import uuid

def flatten_dir(root_dir):
    dirs = []
    for r, d, f in os.walk(root_dir):
        for fname in f:
            old = os.path.join(r, fname)

            new = os.path.join(root_dir, "_".join((r.replace("/", "_"), fname)).replace(f"{root_dir}", "")) 

            os.rename(old, new)
        for dirname in d:
            dirs.append(os.path.join(r, dirname))

    dirs = sorted(dirs, key=len)
    dirs.reverse()
    for d in dirs:
        os.rmdir(d)


corpus_root = "e1gen-v2-final-corp-no-kquery"
target_fuzzers = os.listdir(corpus_root)
target_fuzzers = [(tf.rsplit('-', 1)[0], tf.rsplit('-', 1)[1], tf) for tf in target_fuzzers]

sp.run("rm -r dest", shell=True)

for (target, fuzzer, orig) in target_fuzzers:
    os.makedirs(f"dest/{target}", exist_ok = True)
    print(f"{corpus_root}/{orig}", f"dest/{target}/{fuzzer}")
    shutil.copytree(f"{corpus_root}/{orig}", f"dest/{target}/{fuzzer}")
    flatten_dir(f"dest/{target}/{fuzzer}")

