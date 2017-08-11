#!/usr/bin/python

import os
import sys
import argparse
import pickle
import math

from nltk.tokenize.moses import MosesTokenizer

"""
Example command to run program:

python entity_linking.py --index_ent ../indexes/entity.pkl --index_reach ../indexes/reachability_2M.pkl \
    --index_names ../indexes/names.pkl --ent_result ../entity_detection/gold-results/test.txt \
    --rel_result ../relation_prediction/results/main-test-results.txt --output ./results
"""
tokenizer = MosesTokenizer()

def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def get_index(index_path):
    print("loading index from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index

def get_query_text(ent_resultpath):
    print("getting query text...")
    lineids = []
    id2query = {}
    with open(ent_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            try:
                lineid = items[0].strip()
                mid = items[1].strip()
                query = items[2].strip()
            except:
                print("ERROR: line does not have 3 items  -->  {}".format(line.strip()))
                continue
            # print("{}   -   {}".format(lineid, query))
            lineids.append(lineid)
            id2query[lineid] = query
    return lineids, id2query

def get_relations(rel_resultpath):
    print("getting relations...")
    lineids = []
    id2rel = {}
    with open(rel_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            lineid = items[0].strip()
            rel = items[1].strip()
            score = items[2].strip()
            # print("{}   -   {}".format(lineid, rel))
            lineids.append(lineid)
            id2rel[lineid] = rel
    return lineids, id2rel

def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)

def calc_tf_idf(query, cand_ent_name, cand_ent_count, num_entities, index_ent):
    query_terms = tokenizer.tokenize(query)
    try:
        doc_tokens = tokenizer.tokenize(cand_ent_name)
    except:
        print("PROBLEM: doc_tokens: {}".format(doc_tokens))
        doc_tokens = [cand_ent_name, cand_ent_name.replace(".", "")]
    common_terms = set(query_terms).intersection(set(doc_tokens))

    # len_intersection = len(common_terms)
    # len_union = len(set(query_terms).union(set(doc_tokens)))
    # tf = len_intersection / len_union
    tf = math.log10(cand_ent_count + 1)
    k1 = 0.5
    k2 = 0.5
    total_idf = 0
    for term in common_terms:
        df = len(index_ent[term])
        diff = max(num_entities - df, 1)
        idf_term = (diff + k1) / (df + k2)
        print("term: {}".format(term))
        print("df: {}".format(df))
        print("num entities: {}".format(num_entities))
        print("idf term: {}".format(idf_term))
        idf = math.log10( idf_term )
        total_idf += idf
    return tf * total_idf


## MAIN FILE - debug
parser = argparse.ArgumentParser(description='Do entity linking')
parser.add_argument('--index_ent', dest='index_ent', action='store', required=True,
                    help='path to the pickle for the inverted entity index')
parser.add_argument('--index_reach', dest='index_reach', action='store', required=True,
                    help='path to the pickle for the graph reachability index')
parser.add_argument('--index_names', dest='index_names', action='store', required=True,
                    help='path to the pickle for the names index')
parser.add_argument('--ent_result', dest='ent_result', action='store', required=True,
                    help='file path to the entity detection results with the query texts')
parser.add_argument('--rel_result', dest='rel_result', action='store', required=True,
                    help='file path to the relation prediction results')
parser.add_argument('--output', dest='output', action='store', required=True,
                    help='directory path to the output of entity linking')

args = parser.parse_args()
print("Index - Entity: {}".format(args.index_ent))
print("Index - Reachability: {}".format(args.index_reach))
print("Index - Names: {}".format(args.index_names))
print("Entity Detection Results: {}".format(args.ent_result))
print("Relation Prediction Results: {}".format(args.rel_result))
print("Output: {}".format(args.output))
print("-" * 80)

if not os.path.exists(args.output):
    os.makedirs(args.output)

index_entpath = args.index_ent
index_reachpath = args.index_reach
index_namespath = args.index_names
ent_resultpath = args.ent_result
rel_resultpath = args.rel_result
outpath = args.output

# outfile = open(os.path.join(outpath, "linking-results.txt"), 'w')
notfound_ent = 0
notfound_c = 0

index_ent = get_index(index_entpath)
index_reach = get_index(index_reachpath)
index_names = get_index(index_namespath)
rel_lineids, id2rel = get_relations(rel_resultpath)
ent_lineids, id2query = get_query_text(ent_resultpath)  # ent_lineids may have some examples missing
num_entities = len(index_names)

for i, lineid in enumerate(rel_lineids):
    if lineid not in ent_lineids:
        notfound_ent += 1
        continue

    pred_relation = www2fb(id2rel[lineid])
    query_text = id2query[lineid].lower()  # lowercase the query
    query_tokens = tokenizer.tokenize(query_text)

    print("lineid: {}, query_text: {}, relation: {}".format(lineid, query_text, pred_relation))
    print("query_tokens: {}".format(query_tokens))

    N = min(len(query_tokens), 3)
    C = []  # candidate entities
    for n in range(N, 0, -1):
        ngrams_set = find_ngrams(query_tokens, n)
        # print("ngrams_set: {}".format(ngrams_set))
        for ngram_tuple in ngrams_set:
            ngram = " ".join(ngram_tuple)
            print("ngram: {}".format(ngram))
            ## PROBLEM! - ngram doesnt exist in index
            cand_mids = index_ent[ngram]  # search entities
            C.extend(cand_mids)
            # print("C: {}".format(C))
        if (len(C) > 0):
            print("early termination...")
            break  # early termination
    print("C[:5]: {}".format(C[:5]))

    # relation correction
    C_pruned = []
    for mid in set(C):
        if mid in index_reach.keys():  # PROBLEM: don't know why this may not exist??
            if pred_relation in index_reach[mid]:
                count_mid = C.count(mid) # count number of times mid appeared in C
                C_pruned.append( (mid, count_mid) )
    print("C_pruned[:5]: {}".format(C_pruned[:5]))

    C_tfidf_pruned = []
    for mid, count_mid in C_pruned:
        try:
            cand_ent_name = index_names[mid]
        except:
            print("WARNING: mid: {} - not in index names.".format(mid))
            continue
        tfidf = calc_tf_idf(query_text, cand_ent_name, count_mid, num_entities, index_ent)
        C_tfidf_pruned.append((mid, tfidf))
    print("C_tfidf_pruned[:10]: {}".format(C_tfidf_pruned[:10]))

    if len(C_tfidf_pruned) == 0:
        print("WARNING: C_tfidf_pruned is empty.")
        notfound_c += 1
        continue

    C_tfidf_pruned.sort(key=lambda t: -t[1])
    pred_ent_mid = C_tfidf_pruned[0][0] # get first entry's mid

    line_to_print = "PRED: {}\t{}\t{}".format(lineid, pred_ent_mid, pred_relation)
    print(line_to_print)

    if (i+1) % 10 == 0:
        break
#     outfile.write(line_to_print + "\n")
#
# outfile.close()

print("Entity Linking done.")
