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
from utils import tokenize_text, www2fb, get_index, strip_accents, find_ngrams, calc_tf_idf, pick_best_name, ins, get_span, get_names

stopwords = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()

def get_query_text(input_sent, questions, ent_model, index2tag):
    sent = tokenizer.tokenize(input_sent.lower())
    example = ins(questions.numericalize(questions.pad([sent]), device=args.gpu, train=False))
    ent_model.eval()
    scores = ent_model(example)
    index_tag = np.transpose(torch.max(scores, 1)[1].cpu().data.numpy())
    tag_array = index2tag[index_tag][0]
    spans = get_span(tag_array)
    query_tokens = []
    for span in spans:
        query_tokens.append(" ".join(sent[span[0]:span[1]]))
    return query_tokens

def get_relation(input_sent, questions, model, index2rel):
    sent = tokenizer.tokenize(input_sent.lower())
    example = ins(questions.numericalize(questions.pad([sent]), device=args.gpu, train=False))
    model.eval()
    scores = model(example)
    # get the predicted relations
    top_scores, top_indices = torch.max(scores, dim=1)  # shape: (batch_size, 1)
    top_index = top_indices.cpu().data.numpy()[0]
    predicted_relation = index2rel[top_index]
    return predicted_relation

class Server():
    def __init__(self):
        index_entpath = "indexes/entity_2M.pkl"
        index_reachpath = "indexes/reachability_2M.pkl"
        index_namespath = "indexes/names_2M.pkl"
        self.index_ent = get_index(index_entpath)
        self.index_names = get_index(index_namespath)
        self.index_reach = get_index(index_reachpath)
    def setup(self):
        args = get_args()
        torch.manual_seed(args.seed)
        if not args.cuda:
            args.gpu = -1
        if torch.cuda.is_available() and not args.cuda:
            print("WARNING: You have CUDA but not using it.")
        if torch.cuda.is_available() and args.cuda:
            torch.cuda.set_device(args.gpu)
            torch.cuda.manual_seed(args.seed)

        # for entity detection
        questions2 = data.Field(lower=True, sequential=True)
        relations2 = data.Field(sequential=False)
        labels = data.Field(sequential=True)
        train2, dev2, test2 = SimpleQADataset.splits(questions2, labels, root='./entity_detection/data')
        questions2.build_vocab(train2, dev2, test2)
        labels.build_vocab(train2, dev2, test2)
        index2tag = np.array(labels.vocab.itos)
        index2word = np.array(questions2.vocab.itos)

        # for relation prediction
        questions = data.Field(lower=True, tokenize=tokenize_text)
        relations = data.Field(sequential=False)
        train, dev, test = SimpleQaRelationDataset.splits(questions, relations, root="./data")
        train_iter, dev_iter, test_iter = SimpleQaRelationDataset.iters(args, questions, relations, train, dev, test, shuffleTrain=False)
        questions.build_vocab(train, dev, test)
        index2rel = np.array(relations.vocab.itos)
    
        if os.path.isfile(args.vector_cache):
            questions.vocab.vectors = torch.load(args.vector_cache)
            questions2.vocab.vectors = torch.load(args.ent_vector_cache)
        else:
            questions.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
            os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
            torch.save(questions.vocab.vectors, args.vector_cache)
            questions2.vocab.load_vectors(wv_dir=args.ent_data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
            os.makedirs(os.path.dirname(args.ent_vector_cache), exist_ok=True)
            torch.save(questions2.vocab.vectors, args.ent_vector_cache)
    
        # set up models from trained data
        state = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))
        state2 = torch.load(args.ent_trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))
    
        config = args
        config.n_embed = len(questions.vocab) # vocab. size / number of embeddings
        config.d_out = len(relations.vocab)
        config.n_cells = config.n_layers
        if config.birnn:
            config.n_cells *= 2
        print(config)
        model = RelationClassifier(config)
    
        config.n_out = len(labels.vocab) # I/in entity  O/out of entity
        print(config)
        ent_model = EntityDetection(config)
    
        if args.word_vectors:
            model.embed.weight.data = questions.vocab.vectors
            ent_model.embed.weight.data = questions2.vocab.vectors
            if args.cuda:
                model.cuda()
                ent_model.cuda()

        model.load_state_dict(state)
        ent_model.load_state_dict(state2)

        self.questions = questions
        self.model = model
        self.ent_model = ent_model
        self.index2rel = index2rel
        self.index2tag = index2tag

    def answer(self, question):
        pred_relation = www2fb(get_relation(question, self.questions, self.model, self.index2rel))
        print(pred_relation)
        
        query_tokens = get_query_text(input_sent, self.questions, self.ent_model, self.index2tag)
        print(query_tokens)
        
        N = min(len(query_tokens), 3)
        print(N)
        
        C = []  # candidate entities
        for n in range(N, 0, -1):
            ngrams_set = find_ngrams(query_tokens, n)
            print("ngrams_set: {}".format(ngrams_set))
            for ngram_tuple in ngrams_set:
                ngram = " ".join(ngram_tuple)
                ngram = strip_accents(ngram)
                # unigram stopwords have too many candidates so just skip over
                if ngram in stopwords:
                    continue
                print("ngram: {}".format(ngram))
                ## PROBLEM! - ngram doesnt exist in index - at test-2592 - KeyError: 'p.a.r.c.e. parce'
                try:
                    cand_mids = self.index_ent[ngram]  # search entities
                except:
                    continue
                C.extend(cand_mids)
                # print("C: {}".format(C))
            if (len(C) > 0):
                print("early termination...")
                break
            break
        print(C)
        
        C_pruned = []
        for mid in set(C):
            if mid in self.index_reach.keys():  # PROBLEM: don't know why this may not exist??
                count_mid = C.count(mid)  # count number of times mid appeared in C
                C_pruned.append((mid, count_mid))
                if pred_relation in self.index_reach[mid]:
                    count_mid = C.count(mid)  # count number of times mid appeared in C
                    C_pruned.append((mid, count_mid))
        print(C_pruned)
        
        num_entities_fbsubset = 1959820  # 2M - 1959820 , 5M - 1972702
        C_tfidf_pruned = []
        for mid, count_mid in C_pruned:
            if mid in self.index_names.keys():
                cand_ent_name = pick_best_name(question, self.index_names[mid])
                tfidf = calc_tf_idf(query_text, cand_ent_name, count_mid, num_entities_fbsubset, self.index_ent)
                C_tfidf_pruned.append((mid, cand_ent_name, tfidf))
        # print("C_tfidf_pruned[:10]: {}".format(C_tfidf_pruned[:10]))
        print(C_tfidf_pruned)
        
        C_tfidf_pruned.sort(key=lambda t: -t[2])
        pred_ent, name_ent, score = C_tfidf_pruned[0]
        print(pred_ent)
        print(name_ent)
        
        # FIXME: store the Freebase graph with the name field
        fb_path = "indexes/fb_graph.pkl"
        fb_graph = get_index(fb_path)
        
        # FIXME: lookup Freebase for object of (ent, rel)
        result_mid = fb_graph[(pred_ent, pred_relation)]
        result_mid = list(result_mid)
        print(result_mid)
        
        # FIXME: lookup Freebase for the name predicate of that object
        result = get_names(fb_graph, result_mid)[0]
        print("Answer: {}".format(result))
        return result
