import os
import sys
import argparse
import pickle
import math
import unicodedata
import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords


index_entpath = "../indexes/entity_2M.pkl"
index_reachpath = "../indexes/reachability_2M.pkl"
index_namespath = "../indexes/names_2M.pkl"
ent_resultpath = "../entity_detection/query-text/val.txt"
rel_resultpath = "../relation_prediction/results/topk-retrieval-valid-hits-3.txt"
outpath = "./tmp/results"


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

def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')


def get_query_text(ent_resultpath):
    print("getting query text...")
    lineids = []
    id2query = {}
    notfound = 0
    with open(ent_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            try:
                lineid = items[0].strip()
                query = items[1].strip()
                # mid = items[2].strip()
            except:
                # print("ERROR: line does not have >2 items  -->  {}".format(line.strip()))
                notfound += 1
                continue
            # print("{}   -   {}".format(lineid, query))
            lineids.append(lineid)
            id2query[lineid] = query
    print("notfound (empty query text): {}".format(notfound))
    return lineids, id2query

def get_relations(rel_resultpath):
    print("getting relations...")
    lineids = []
    id2rels = {}
    with open(rel_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            lineid = items[0].strip()
            rel = www2fb(items[1].strip())
            label = items[2].strip()
            score = items[3].strip()
            # print("{}   -   {}".format(lineid, rel))
            if lineid in id2rels.keys():
                id2rels[lineid].append( (rel, label, score) )
            else:
                id2rels[lineid] = [(rel, label, score)]
                lineids.append(lineid)
    return lineids, id2rels


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

index_ent = get_index(index_entpath)
num_entities_fbsubset = 1959820  # 2M - 1959820 , 5M - 1972702

index_names = get_index(index_namespath)
index_reach = get_index(index_reachpath)


def pick_best_name(question, names_list):
    best_score = None
    best_name = None
    for name in names_list:
        score =  fuzz.ratio(name, question)
        if best_score == None or score > best_score:
            best_score = score
            best_name = name

    return best_name


rel_lineids, id2rel = get_relations(rel_resultpath)
ent_lineids, id2query = get_query_text(ent_resultpath)  # ent_lineids may have some examples missing


def get_questions(datapath):
    print("getting questions...")
    id2question = {}
    with open(datapath, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            lineid = items[0].strip()
            sub = items[1].strip()
            pred = items[2].strip()
            obj = items[3].strip()
            question = items[4].strip()
            # print("{}   -   {}".format(lineid, question))
            if lineid.startswith("valid"):
                id2question[lineid] = (sub, pred, question)
    return id2question

datapath = "../data/SimpleQuestions_v2_modified/all.txt"
id2question = get_questions(datapath)

def fuzzy_match_score(name, question):
    score =  fuzz.ratio(name, question) / 100.0
    return score

notfound_ent = 0
notfound_c = 0
notfound_c_lineids = []
notfound_ent = 0
notcorrect_ent_lineids = []
candidate_mids = {}
HITS_TOP_ENTITIES = 5

for i, lineid in enumerate(rel_lineids):
    if lineid not in ent_lineids:
        notfound_ent += 1
        continue

    if i % 1000 == 0:
        print("line {}".format(i))

    question = id2question[lineid]
    query_text = id2query[lineid].lower()  # lowercase the query
    query_tokens = tokenize_text(query_text)

    # print("lineid: {}, query_text: {}, relation: {}".format(lineid, query_text, pred_relation))
    # print("query_tokens: {}".format(query_tokens))

    N = min(len(query_tokens), 3)
    C = []  # candidate entities
    for n in range(N, 0, -1):
        ngrams_set = find_ngrams(query_tokens, n)
        # print("ngrams_set: {}".format(ngrams_set))
        for ngram_tuple in ngrams_set:
            ngram = " ".join(ngram_tuple)
            ngram = strip_accents(ngram)
            # unigram stopwords have too many candidates so just skip over
            if ngram in stopwords:
                continue
            # print("ngram: {}".format(ngram))
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
            count_mid = C.count(mid)  # count number of times mid appeared in C
            C_pruned.append((mid, count_mid))

    C_tfidf_pruned = []
    for mid, count_mid in C_pruned:
        if mid in index_names.keys():
            cand_ent_name = pick_best_name(question[2], index_names[mid])
            score = calc_tf_idf(query_text, cand_ent_name, count_mid, num_entities_fbsubset, index_ent)
#             score = fuzzy_match_score(cand_ent_name, query_text)
            C_tfidf_pruned.append((mid, cand_ent_name, score))
    # print("C_tfidf_pruned[:10]: {}".format(C_tfidf_pruned[:10]))

    if len(C_tfidf_pruned) == 0:
        #print("WARNING: C_tfidf_pruned is empty.")
        notfound_c_lineids.append(lineid)
        notfound_c += 1
        continue

    C_tfidf_pruned.sort(key=lambda t: -t[2])
    cand_mids = C_tfidf_pruned[:HITS_TOP_ENTITIES]

    candidate_mids[lineid] = cand_mids


pickle.dump(candidate_mids, open('cand_mids.pkl', 'wb'))
