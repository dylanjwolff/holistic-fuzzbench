
def count_unexplored_branches(fpath):
    with open(fpath) as f:
        lines = f.readlines()

        trues = set()
        falses = set()
        for line in lines:
            addr = line.split(':')[0]
            if "TRUE" in line:
                trues.add(addr)
            elif "FALSE" in line:
                falses.add(addr)
            else:
                raise Exception("All lines should contatin TRUE or FALSE")

        return(len(trues ^ falses))

def count_unique_reached(fpath):
    with open(fpath) as f:
        lines = f.readlines()
        return len(set([line.split(':')[0] for line in lines]))


if __name__ == "__main__":
    print(count_unexplored_branches("out.txt"))
    print(count_unique_reached("out.txt"))
