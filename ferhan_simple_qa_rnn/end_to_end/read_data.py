fname = "ent-results/out.BruteSearch.2M.h20.txt"

hits = 20
lines = [line.rstrip('\n') for line in open(fname)]
lines_to_skip = hits + 3

for i in range(0, len(lines), lines_to_skip):
    end_index = i + lines_to_skip - 1
    if not lines[end_index].startswith("-----------------"):
        print("ERROR!!")

    curr_data = lines[i:end_index-1]
    # print(curr_data)

    lineid, question, gold_mid = curr_data[0].split(" %% ")
    print("{} - {} - {}".format(lineid, question, gold_mid))

    for cand_ent in curr_data[1:]:
       mid, name, score = cand_ent.split(" %% ")
       # print("{} - {} - {}".format(mid, name, score))

