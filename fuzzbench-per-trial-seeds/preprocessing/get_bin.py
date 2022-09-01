import os
import subprocess as sp

bin_names = {"openssl_x509": "./x509",
             "mbedtls_fuzz_dtlsclient": "./fuzz_dtlsclient",
             "libpcap_fuzz_both": "./fuzz_both"}


sut = os.getenv("BIN")
benchmark = os.getenv("TARGET")

if sut is None or sut == "":
    if benchmark is not None and benchmark in bin_names:
        sut = bin_names[benchmark]
        print(sut)
        sp.run(f"mv {sut} sut", shell=True)
        exit()

    p = sp.run('find . -type f -executable -maxdepth 1 | grep -v "afl-fuzz" | grep -v "entry.py"', shell=True, capture_output=True)
    bins = p.stdout.split()
    if len(bins) > 1:
        raise Exception(f"unable to pick out binary from {bins}")
    sut = bins[0].decode("utf-8")


print(sut)
sp.run(f"mv {sut} sut", shell=True)


