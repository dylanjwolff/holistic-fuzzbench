import subprocess as sp
import os

sut = "sut"

sp.run("e9compile patches/prog-stats.c", shell=True)

instrums = os.getenv("INSTRMS").split(",")

all_elapsed = []
all_sizes = []

for inst in instrums:
    sp.run(f"./patches/{inst}.sh {sut} {inst}", shell=True)
