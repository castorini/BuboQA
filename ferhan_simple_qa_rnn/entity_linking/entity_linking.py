#!/usr/bin/python

import os
import sys
import argparse
import pickle
import math

def get_index(index_path):
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index

def get_query_text(ent_resultpath):
    lineids = []
    id2query = {}
    with open(ent_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            lineid = items[0].strip()
            query = items[1].strip()
            lineids.append(lineid)
            id2query[lineid] = query
    return lineids, id2query

def get_relations(rel_resultpath):
    lineids = []
    id2rel = {}
    with open(rel_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            lineid = items[0].strip()
            rel = items[1].strip()
            score = items[2].strip()
            lineids.append(lineid)
            id2rel[lineid] = rel
    return lineids, id2rel

def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)

def get_tf_idf(query, cand_ent_name, num_entities, index_ent):
    query_terms = query.split()
    doc_tokens = cand_ent_name.split()
    common_terms = set(query_terms).intersection(set(doc_tokens))

    # len_intersection = len(common_terms)
    # len_union = len(set(query_terms).union(set(doc_tokens)))
    # tf = len_intersection / len_union

    total_idf = 0
    for term in common_terms:
        df = len(index_ent[term])
        k1 = 0.5
        k2 = 0.5
        idf = math.log10( (num_entities - df + k1) / (df + k2) )
        total_idf += idf
    return total_idf

def entity_linking(index_entpath, index_reachpath, index_namespath, ent_resultpath, rel_resultpath, outpath):
    outfile = open(os.path.join(outpath, "linking-results.txt"), 'w')

    index_ent = get_index(index_entpath)
    index_reach = get_index(index_reachpath)
    index_names = get_index(index_namespath)
    rel_lineids, id2rel = get_relations(rel_resultpath)
    ent_lineids, id2query = get_query_text(ent_resultpath) # ent_lineids may have some examples missing
    num_entities = len(index_names)

    for lineid in rel_lineids:
        if lineid not in ent_lineids:
            continue

        pred_relation = id2rel[id]
        query_text = id2query[id].lower()
        query_tokens = query_text.split()
        N = min(len(query_tokens), 3)
        C = [] # candidate entities
        for n in range(N, 0, -1):
            ngrams_set = find_ngrams(query_tokens, n)
            for ngram_tuple in ngrams_set:
                ngram = " ".join(ngram_tuple)
                cand_mids = index_ent[ngram] # search entities
                C.extend(cand_mids)
            if (len(C) > 0):
                break  # early termination

        C_tfidf = []
        for mid in C:
            cand_ent_name = index_names[mid].lower()
            tfidf = get_tf_idf(query_text, cand_ent_name, num_entities, index_ent)
            C_tfidf.append( (mid, tfidf) )

        # relation correction
        C_tfidf_pruned = []
        for mid, tfidf in C_tfidf:
            if pred_relation in index_reach[mid]:
                C_tfidf_pruned.append( (mid, tfidf) )

        if len(C_tfidf_pruned) == 0:
            continue

        C_tfidf_pruned.sort(key=lambda t: -t[1])
        pred_ent_mid = C_tfidf_pruned[0][0]

        line_to_print = "{}\t{}\t{}".format(lineid, pred_ent_mid, pred_relation)
        print(line_to_print)
        outfile.write(line_to_print + "\n")

    outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do entity linking')
    parser.add_argument('--index_ent', dest='index_ent', action='store', required = True,
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

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    entity_linking(args.index_ent, args.index_reach, args.index_names, args.ent_result, args.rel_result, args.output)
    print("Entity Linking done.")
