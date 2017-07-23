fname = "ent-results/out.BruteSearch.2M.h20.txt"

hits = 20
lines = [line.rstrip('\n') for line in open(fname)]
lines_to_skip = hits + 3

for i in range(lines_to_skip, len(lines), lines_to_skip):
    if not lines[i-1].startswith("-----------------"):
        print("ERROR!!")

    curr_data = lines[:i-2]

    lineid, question, gold_mid = curr_data[0].split(" %% ")
    print("{} - {} - {}".format(lineid, question, gold_mid))

    for cand_ent in curr_data[1:]:
        mid, name, score = cand_ent.split(" %% ")
        # print("{} - {} - {}".format(mid, name, score))

    break