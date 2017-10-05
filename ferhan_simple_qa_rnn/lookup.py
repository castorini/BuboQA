# when was albert einstein born?

# PREDICTED RELATION:
# fb:people.person.place_of_birth
# QUERY TOKENS:
# ['albert einstein']
# UNKNOWN
# BuboQA is up
#  * Running on http://127.0.0.1:4001/ (Press CTRL+C to quit)
# How long is the Nile river?
# PREDICTED RELATION:
# fb:geography.river.mouth
# QUERY TOKENS:
# ['nile river']
# Mediterranean Sea
# 127.0.0.1 - - [11/Sep/2017 09:57:45] "GET /ask/How%20long%20is%20the%20Nile%20river%3F HTTP/1.1" 200 -
# What is a region that dead combo was released in?
# PREDICTED RELATION:
# fb:music.release.region
# QUERY TOKENS:
# ['dead combo']
# UNKNOWN
# 127.0.0.1 - - [11/Sep/2017 09:57:51] "GET /ask/What%20is%20a%20region%20that%20dead%20combo%20was%20released%20in%3F HTTP/1.1" 200 -
# what is the book e about
# PREDICTED RELATION:
# fb:book.written_work.subjects
# QUERY TOKENS:
# ['e']
# UNKNOWN
import os
import sys
import torch
import pickle
import math
import unicodedata
import pandas as pd
import numpy as np

from args import get_args, get_ent_args
from torchtext import data

from entity_detection.simple_qa_ner import SimpleQADataset
from relation_prediction.simple_qa_relation import SimpleQaRelationDataset
from fuzzywuzzy import fuzz
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
from relation_prediction.model import RelationClassifier
from entity_detection.model import EntityDetection
from utils import tokenize_text, www2fb, get_index, strip_accents, find_ngrams, calc_tf_idf, pick_best_name, ins, get_span, get_names

def answer(relation, tokens):
    fb_path = "indexes/fb_graph.pkl"
    index_entpath = "indexes/entity_2M.pkl"
    index_reachpath = "indexes/reachability_2M.pkl"
    index_namespath = "indexes/names_2M.pkl"
    fb_path = "indexes/fb_graph.pkl"
    index_ent = get_index(index_entpath)
    index_names = get_index(index_namespath)
    index_reach = get_index(index_reachpath)
    fb_graph = get_index(fb_path)
    pred_relation = relation
    query_tokens = tokens

    N = min(len(query_tokens), 3)

    C = []  # candidate entities
    for n in range(N, 0, -1):
        ngrams_set = find_ngrams(query_tokens, n)
        for ngram_tuple in ngrams_set:
            ngram = " ".join(ngram_tuple)
            ngram = strip_accents(ngram)
            # unigram stopwords have too many candidates so just skip over
            if ngram in stopwords:
                continue
            ## PROBLEM! - ngram doesnt exist in index - at test-2592 - KeyError: 'p.a.r.c.e. parce'
            try:
                cand_mids = index_ent[ngram]  # search entities
            except:
                continue
            C.extend(cand_mids)
        if (len(C) > 0):
            break
        break

    C_pruned = []
    for mid in set(C):
        if mid in index_reach.keys():  # PROBLEM: don't know why this may not exist??
            count_mid = C.count(mid)  # count number of times mid appeared in C
            C_pruned.append((mid, count_mid))
            if pred_relation in index_reach[mid]:
                count_mid = C.count(mid)  # count number of times mid appeared in C
                C_pruned.append((mid, count_mid))

    num_entities_fbsubset = 1959820  # 2M - 1959820 , 5M - 1972702
    C_tfidf_pruned = []
answer(sys.argv[1], list(sys.argv[2]))
