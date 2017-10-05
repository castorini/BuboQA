import os
import sys
import torch
import pickle
import math
import unicodedata
import pandas as pd
import numpy as np

from args import get_args
from torchtext import data

from entity_detection.simple_qa_ner import SimpleQADataset
from relation_prediction.simple_qa_relation import SimpleQaRelationDataset
from fuzzywuzzy import fuzz
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
from relation_prediction.model import RelationClassifier
from entity_detection.model import EntityDetection

stopwords = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()

# UTILS
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

def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)

def calc_tf_idf(query_terms, cand_ent_name, cand_ent_count, num_entities, index_ent):
    #query_terms = tokenize_text(query)
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
        idf = math.log10((num_entities - df + k1) / (df + k2))
        total_idf += idf
    return tf * total_idf

def pick_best_name(question, names_list):
    best_score = None
    best_name = None
    for name in names_list:
        score = fuzz.ratio(name, question)
        if best_score == None or score > best_score:
            best_score = score
            best_name = name
    return best_name

class ins(object):
    def __init__(self, question):
      self.question = question

def get_span(label):
    span = []
    st = 0
    en = 0
    flag = False
    for k, l in enumerate(label):
        if l == 'I' and flag == False:
            st = k
            flag = True
        if l != 'I' and flag == True:
            flag = False
            en = k
            span.append((st, en))
            st = 0
            en = 0
    if st != 0 and en == 0:
        en = k
        span.append((st, en))
    return span

def get_names(fb_graph, mids):
    names = []
    for mid in mids:
        key1 = (mid, 'fb:type.object.name')
        key2 = (mid, 'fb:common.topic.alias')
        if key1 in fb_graph:
            names.extend( fb_graph[key1] )
        if key2 in fb_graph:
            names.extend( fb_graph[key2] )
    names.sort(key = lambda s: -len(s))
    return names

