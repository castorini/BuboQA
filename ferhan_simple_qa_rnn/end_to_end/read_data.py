import pickle
import pandas as pd

##----------------------
## entity linking results
print("entity linking pass...")

fname = "ent-results/train.BruteSearch.2M.h20.txt"
hits = 20
lines = [line.rstrip('\n') for line in open(fname)]
lines_to_skip_ent = hits + 3
line_ids_ent_found = set()

for i in range(0, len(lines), lines_to_skip_ent):
    end_index = i + lines_to_skip_ent - 1
    curr_data = lines[i:end_index-1]

    lineid, question, gold_mid = curr_data[0].split(" %% ")
    found = "NOT"
    if lines[end_index - 1].startswith("FOUND"):
        found = "FOUND"
        line_ids_ent_found.add(lineid)


##----------------------
## relation prediction results
print("relation prediction pass...")

def convert_line_id(id):
    if id.startswith("train"):
        return "tr{}".format(id.split("train-")[1])
    elif id.startswith("valid"):
        return "va{}".format(id.split("valid-")[1])
    return "te{}".format(id.split("test-")[1])

fname = "rel-results/train-hits-3.txt"

hits = 3
lines = [line.rstrip('\n') for line in open(fname)]
lines_to_skip_rel = hits + 3
lines_rel_needed = []
line_ids_rel_found = set()
total = 0

for i in range(0, len(lines), lines_to_skip_rel):
    total += 1
    end_index = i + lines_to_skip_rel - 1
    curr_data = lines[i:end_index-1]

    lineid, question, gold_rel = curr_data[0].split(" %%%% ")
    lineid = convert_line_id(lineid) # match with entity dataset for now

    if (lineid not in line_ids_ent_found):
        continue

    found = "NOT"
    if lines[end_index - 1].startswith("FOUND"):
        found = "FOUND"
        line_ids_rel_found.add(lineid)

    lines_rel_needed.append(lines[i:end_index+1])


##----------------------
## map of possible entity - list of relations
print("loading the map...")

def www2fb(in_str):
    if in_str.startswith("http://rdf.freebase.com/ns/"):
        return 'fb:%s' % (in_str.split('http://rdf.freebase.com/ns/')[-1].replace('/', '.'))
    elif in_str.startswith("www.freebase.com/"):
        return 'fb:%s' % (in_str.split("www.freebase.com/")[-1].replace('/', '.'))
    elif in_str.startswith("http://www.w3.org/2000/01/rdf-schema#label"):
        return 'w3:rdf-schema#label'
    elif in_str.startswith("http://rdf.freebase.com/key/"):
        return 'fbkey:%s' % (in_str.split('http://rdf.freebase.com/key/')[-1].replace('/', '.'))
    return in_str

def clean_uri(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return clean_uri(uri[1:-1])
    elif uri.startswith("\"") and uri.endswith("\""):
        return clean_uri(uri[1:-1])
    return uri

# datapath = "augmented-fb-subsets/2M.txt"
# ent_to_possible_rels = {}
#
# with open(datapath, 'r') as f:
#     for i, line in enumerate(f):
#         if i % 1000000 == 0:
#             print("line: {}".format(i))
#         items = line.strip().split("\t")
#         if len(items) != 4:
#             print("ERROR: line - {}".format(line))
#         subject = www2fb(clean_uri(items[0]))
#         predicate = www2fb(clean_uri(items[1]))
#
#         # print("{} - {}".format(subject, predicate))
#
#         if subject in ent_to_possible_rels.keys():
#             ent_to_possible_rels[subject].add(predicate)
#         else:
#             ent_to_possible_rels[subject] = set([predicate])
#
# with open("ent_to_possible_rels.pkl", 'wb') as f:
#     pickle.dump(ent_to_possible_rels, f)

with open('ent_to_possible_rels.pkl', 'rb') as f:
    ent_to_possible_rels = pickle.load(f)
print("ent_to_possible_rels length: {}".format(len(ent_to_possible_rels)))

##----------------------
## 2nd pass - entity linking results
print("2nd entity linking pass...")

fname = "ent-results/train.BruteSearch.2M.h20.txt"
lines = [line.rstrip('\n') for line in open(fname)]
data = []
count = 0

for i in range(0, len(lines), lines_to_skip_ent):
    end_index_ent = i + lines_to_skip_ent - 1
    ent_data_block = lines[i:end_index_ent-1]

    lineid, question, gold_mid = ent_data_block[0].split(" %% ")
    gold_mid = "fb:{}".format(gold_mid)

    if (lineid not in line_ids_rel_found):
        continue

    if not lines[end_index_ent - 1].startswith("FOUND"):
        continue

    ### means this line was found in ent and rel
    count += 1
    rel_data_block = lines_rel_needed.pop(0)
    _, _, gold_rel = rel_data_block[0].split(" %%%% ")
    gold_rel = www2fb(gold_rel)

    for cand_ent in set(ent_data_block[1:]):
       ent_mid, ent_name, ent_score = cand_ent.split(" %% ")
       ent_mid = "fb:{}".format(ent_mid)
       for cand_rel in set(rel_data_block[1:-2]):
           rel_name, rel_score = cand_rel.split(" %%%% ")
           rel_name = www2fb(rel_name)
           # trim the impossible combinations
           if (rel_name not in ent_to_possible_rels[ent_mid]):
               continue
           label = 0
           if ent_mid == gold_mid and rel_name == gold_rel:
               label = 1
           # print("{} - label {} - {} - {} - {} - {} - {}".format(lineid, label, ent_mid, rel_name, ent_name, ent_score, rel_score))
           example = [lineid, question, ent_mid, ent_name, rel_name, ent_score, rel_score, label]
           data.append(example)

print("total: {}".format(total))
print("count: {}".format(count))
col_names = pd.Index(["lineid", "question", "ent_mid", "ent_name", "rel_name", "ent_score", "rel_score", "label"], name="columns")
df = pd.DataFrame(data, columns=col_names)
df.to_pickle("end_df.pkl")


