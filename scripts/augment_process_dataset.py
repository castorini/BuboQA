#!/usr/bin/python

import os
import sys
import argparse
import pickle
import logging
# from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
from nltk.tokenize.treebank import TreebankWordTokenizer
from util import www2fb, clean_uri, processed_text
tokenizer = TreebankWordTokenizer()
logger = logging.getLogger()
logger.disabled = True

def get_index(index_path):
    print("loading index from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index

def pick_best_name(question, names_list):
    # best_score = None
    # best_name = None
    # for name in names_list:
    #     score =  fuzz.ratio(name, question)
    #     if best_score == None or score > best_score:
    #         best_score = score
    #         best_name = name
    
    best_name, score = process.extractOne(question, names_list)
    return best_name

def get_indices(src_list, pattern_list):
    indices = None
    for i in range(len(src_list)):
        match = 1
        for j in range(len(pattern_list)):
            if src_list[i+j] != pattern_list[j]:
                match = 0
                break
        if match:
            indices = range(i, i + len(pattern_list))
            break
    return indices

be = ["is", "was", "'s", "were", "are", "'re"]

def reverseLinking(question, names_list):
    # get question tokens
    tokens = question.split()
    # init default value of returned variables
    label = ["O"] * len(tokens)
    exact_match = False
    text_attention_indices = None
    for res in sorted(names_list, key=lambda name: len(name), reverse=True):
        pattern = r'(^|\s)(%s)($|\s)' % (re.escape(res))
        if re.search(pattern, question):
            text_attention_indices = get_indices(tokens, res.split())
            break
    if text_attention_indices != None:
        exact_match = True
        for i in text_attention_indices:
            label[i] = 'I'
    else:
        if names_list == None or len(names_list) == 0:
            return '<UNK>', label, exact_match
        try:
            v, score = process.extractOne(question, names_list)
        except:
            print(question, names_list)
            exit()
        v = tokenizer.tokenize(v)
        for w in v:
            result = process.extractOne(w, tokens, score_cutoff=85)
            if result == None:
                continue
            else:
                word = result[0]
                label[tokens.index(word)] = 'I'
        # Manually correct some error
        # 'is' or 'was' is often matched and labeled as 'I'
        if len(tokens) > 1 and (tokens[1] in be) and label[1] == 'I':
            label[1] = 'O'
        if len(tokens) > 2 and (tokens[2] in be) and label[2] == 'I':
            label[1] = 'O'
        # Smoothing the label (like 'I I I O I I I')
        start = end = -1
        start_len = end_len = 0
        for l in range(len(label)):
            if label[l] == 'I':
                start = l
                start_len += 1
            if label[l] == 'O' and start != -1:
                start = l
                break
        if start != -1:
            for l in range(len(label)):
                if label[len(label) - l - 1] == 'I':
                    end = len(label) - l
                    end_len += 1
                if label[len(label) - l - 1 ] == 'O' and end != -1:
                    end = len(label) - l
                    break
        if start != -1 and end != -1 and start < end:
            if end-start<2:
                for l in range(start, end):
                    label[l] = 'I'
            else:
                if start_len <= end_len:
                    for l in range(start-start_len, start):
                        label[l] = 'O'
                else:
                    for l in range(end, end+end_len):
                        label[l] = 'O'
    entity = []
    for l, t in zip(label, tokens):
        if l == 'I':
            entity.append(t)
    entity = " ".join(entity)
    label = " ".join(label)
    return entity, label, exact_match

def augment_dataset(datadir, index_namespath, outdir):
    names_map = get_index(index_namespath)
    skipped = 0
    allpath = os.path.join(outdir, "all.txt")
    outallfile = open(allpath, 'w')
    print("creating new datasets...")
    files = [("annotated_fb_data_train", "train"), ("annotated_fb_data_valid", "valid"), ("annotated_fb_data_test", "test")]
    for f_tuple in files:
        f = f_tuple[0]
        fname = f_tuple[1]
        fpath = os.path.join(datadir, f + ".txt")
        fpath_numbered = os.path.join(outdir, fname + ".txt")
        total_exact_match = 0
        total = 0
        outfile = open(fpath_numbered, 'w')
        print("reading from {}".format(fpath))
        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0:
                    print("line: {}".format(i))

                items = line.strip().split("\t")
                if len(items) != 4:
                    print("ERROR: line - {}".format(line))
                    sys.exit(0)
                total += 1
                lineid = "{}-{}".format(fname, (i + 1))
                subject = www2fb(items[0])
                predicate = www2fb(items[1])
                object = www2fb(items[2])
                question = processed_text(items[3])

                if subject not in names_map.keys():
                    skipped += 1
                    print("lineid {} - name not found. {} skipping question.".format(lineid, subject))
                    continue

                cand_entity_names = names_map[subject]
                #entity_name = pick_best_name(question, cand_entity_names)

                entity_name, label, exact_match = reverseLinking(question, cand_entity_names)
                if exact_match:
                    total_exact_match += 1

                line_to_print = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(lineid, subject, entity_name, predicate, object, question, label)
                # print(line_to_print)
                outfile.write(line_to_print + "\n")
                outallfile.write(line_to_print + "\n")

        print("wrote to {}".format(fpath_numbered))
        print("Exact Match Entity : {} out of {} : {}".format(total_exact_match, total, total_exact_match/total))
        outfile.close()

    print("wrote to {}".format(allpath))
    print("skipped # questions: {}".format(skipped))
    outallfile.close()
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment dataset with line ids, shorted names, entity names')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required = True,
                        help='path to the dataset directory - contains train, valid, test files')
    parser.add_argument('-i', '--index_names', dest='index_names', action='store', required=True,
                        help='path to the pickle for the names index')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output directory for new dataset')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Index - Names: {}".format(args.index_names))
    print("Output: {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    augment_dataset(args.dataset, args.index_names, args.output)
    print("Augmented dataset with line ids, shorted names, entity names.")
