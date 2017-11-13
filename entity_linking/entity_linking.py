#!/usr/bin/python

import os
import sys
import argparse
import pickle
import math

from fuzzywuzzy import fuzz
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords

"""
Example command to run program:
Gold query text:
python entity_linking.py -t ../data/SimpleQuestions_v2_modified/test.txt --index_ent ../indexes/entity_2M.pkl --index_reach ../indexes/reachability_2M.pkl \
    --index_names ../indexes/names_2M.pkl --ent_result ../entity_detection/gold-query-text/test.txt \
    --rel_result ../relation_prediction/results/main-test-results.txt --output ./gold-results
    
Predicted query text:
python entity_linking.py -t ../data/SimpleQuestions_v2_modified/test.txt --index_ent ../indexes/entity_2M.pkl --index_reach ../indexes/reachability_2M.pkl \
    --index_names ../indexes/names_2M.pkl --ent_result ../entity_detection/query-text/test.txt \
    --rel_result ../relation_prediction/results/main-test-results.txt --output ./results
"""
tokenizer = TreebankWordTokenizer()
stopwords = set(stopwords.words('english'))

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return tokens

def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def get_index(index_path):
    print("loading index from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index

def get_ground_truth(datapath):
    print("getting questions...")
    id2truth = {}
    with open(datapath, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            lineid = items[0].strip()
            sub = items[1].strip()
            pred = items[2].strip()
            question = items[4].strip()
            # print("{}   -   {}".format(lineid, question))
            id2truth[lineid] = (sub, pred, question)
    return id2truth

def pick_best_name(question, names_list):
    best_score = None
    best_name = None
    for name in names_list:
        score =  fuzz.ratio(name, question)
        if best_score == None or score > best_score:
            best_score = score
            best_name = name

    return best_name

def get_query_text(ent_resultpath):
    print("getting query text...")
    lineids = []
    id2query = {}
    with open(ent_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            try:
                lineid = items[0].strip()
                query = items[1].strip()
                # mid = items[2].strip()
            except:
                print("ERROR: line does not have >2 items  -->  {}".format(line.strip()))
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
    query_terms = tokenize_text(query)
    doc_tokens = tokenize_text(cand_ent_name)
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
        idf = math.log10( (num_entities - df + k1) / (df + k2) )
        total_idf += idf
    return tf * total_idf

def link_one_question(lineid, question, query_text, pred_relation, num_entities_fbsubset,
                                                    index_ent, index_reach, index_names):
    query_tokens = tokenize_text(query_text)

    print("lineid: {}, query_text: {}, relation: {}".format(lineid, query_text, pred_relation))
    # print("query_tokens: {}".format(query_tokens))

    N = min(len(query_tokens), 3)
    C = []  # candidate entities
    for n in range(N, 0, -1):
        ngrams_set = find_ngrams(query_tokens, n)
        # print("ngrams_set: {}".format(ngrams_set))
        for ngram_tuple in ngrams_set:
            ngram = " ".join(ngram_tuple)
            # unigram stopwords have too many candidates so just skip over
            if ngram in stopwords:
                continue
            # print("ngram: {}".format(ngram))
            ## PROBLEM! - ngram doesnt exist in index - at test-2592 - KeyError: 'p.a.r.c.e. parce'
            try:
                cand_mids = index_ent[ngram]  # search entities
            except:
                continue
            C.extend(cand_mids)
            # print("C: {}".format(C))
        if (len(C) > 0):
            # print("early termination...")
            break
    # print("C[:5]: {}".format(C[:5]))

    # relation correction
    C_pruned = []
    for mid in set(C):
        if mid in index_reach.keys():  # PROBLEM: don't know why this may not exist??
            if pred_relation in index_reach[mid]:
                count_mid = C.count(mid)  # count number of times mid appeared in C
                C_pruned.append((mid, count_mid))
    # print("C_pruned[:5]: {}".format(C_pruned[:5]))

    C_tfidf_pruned = []
    for mid, count_mid in C_pruned:
        if mid in index_names.keys():
            cand_ent_name = pick_best_name(question, index_names[mid])
            tfidf = calc_tf_idf(query_text, cand_ent_name, count_mid, num_entities_fbsubset, index_ent)
            C_tfidf_pruned.append((mid, tfidf))
    # print("C_tfidf_pruned[:10]: {}".format(C_tfidf_pruned[:10]))

    if len(C_tfidf_pruned) == 0:
        print("WARNING: C_tfidf_pruned is empty.")
        return None

    C_tfidf_pruned.sort(key=lambda t: -t[1])
    pred_ent_mid = C_tfidf_pruned[0][0]  # get first entry's mid

    return pred_ent_mid

def entity_linking(test_datapath, index_entpath, index_reachpath, index_namespath, ent_resultpath, rel_resultpath, outpath):
    outfile = open(os.path.join(outpath, "linking-results.txt"), 'w')
    notfound_ent = 0
    notfound_c = 0

    index_ent = get_index(index_entpath)
    index_reach = get_index(index_reachpath)
    index_names = get_index(index_namespath)

    id2truth = get_ground_truth(test_datapath)
    rel_lineids, id2rel = get_relations(rel_resultpath)
    ent_lineids, id2query = get_query_text(ent_resultpath)  # ent_lineids may have some examples missing
    num_entities_fbsubset = 1959820  # 2M - 1959820 , 5M - 1972702

    for i, lineid in enumerate(rel_lineids):
        if lineid not in ent_lineids:
            notfound_ent += 1
            continue

        truth_ent, truth_rel, question = id2truth[lineid]
        pred_relation = www2fb(id2rel[lineid])
        query_text = id2query[lineid].lower()  # lowercase the query

        pred_ent_mid = link_one_question(lineid, question, query_text, pred_relation, num_entities_fbsubset,
                                                    index_ent, index_reach, index_names)
        if pred_ent_mid == None:
            notfound_c += 1
            continue

        line_to_print = "{}\t{}\t{}".format(lineid, pred_ent_mid, pred_relation)
        print("PRED: " + line_to_print)

        # if (i+1) % 10 == 0:
        #     break
        outfile.write(line_to_print + "\n")

    print("notfound_ent : {}".format(notfound_ent))
    print("notfound_c : {}".format(notfound_c))
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do entity linking')
    parser.add_argument('-t', '--test_data', dest='test_data', action='store', required=True,
                        help='path to the NUMBERED dataset test.txt file')
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
    print("Test Data: {}".format(args.test_data))
    print("Index - Entity: {}".format(args.index_ent))
    print("Index - Reachability: {}".format(args.index_reach))
    print("Index - Names: {}".format(args.index_names))
    print("Entity Detection Results: {}".format(args.ent_result))
    print("Relation Prediction Results: {}".format(args.rel_result))
    print("Output: {}".format(args.output))
    print("-" * 80)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    entity_linking(args.test_data, args.index_ent, args.index_reach, args.index_names, args.ent_result, args.rel_result, args.output)

    print("Entity Linking done.")
