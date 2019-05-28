#!/usr/bin/python
import os
import sys
import argparse
import pickle
import logging
from fuzzywuzzy import process, fuzz
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



def get_ngram(tokens):
    ngram = []
    for i in range(1, len(tokens)+1):
        for s in range(len(tokens)-i+1):
            ngram.append((" ".join(tokens[s: s+i]), s, i+s))
    return ngram


def reverseLinking(sent, text_candidate):
    tokens = sent.split()
    label = ["O"] * len(tokens)
    text_attention_indices = None
    exact_match = False

    if text_candidate is None or len(text_candidate) == 0:
        return '<UNK>', " ".join(label), exact_match

    # sorted by length
    for text in sorted(text_candidate, key=lambda x:len(x), reverse=True):
        pattern = r'(^|\s)(%s)($|\s)' % (re.escape(text))
        if re.search(pattern, sent):
            text_attention_indices = get_indices(tokens, text.split())
            break
    if text_attention_indices != None:
        exact_match = True
        for i in text_attention_indices:
            label[i] = 'I'
    else:
        try:
            v, score = process.extractOne(sent, text_candidate, scorer=fuzz.partial_ratio)
        except:
            print("Extraction Error with FuzzyWuzzy : {} || {}".format(sent, text_candidate))
            return '<UNK>', " ".join(label), exact_match
        v = v.split()
        n_gram_candidate = get_ngram(tokens)
        n_gram_candidate = sorted(n_gram_candidate, key=lambda x: fuzz.ratio(x[0], v), reverse=True)
        top = n_gram_candidate[0]
        for i in range(top[1], top[2]):
            label[i] = 'I'
    entity_text = []
    for l, t in zip(label, tokens):
        if l == 'I':
            entity_text.append(t)
    entity_text = " ".join(entity_text)
    label = " ".join(label)
    return entity_text, label, exact_match

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

                if names_map.get(subject) is None:
                    if fname != "test":
                        skipped += 1
                        print("lineid {} - name not found. {} Skip".format(lineid, subject))
                        continue
                    else:
                        cand_entity_names = []
                        print("lineid {} - name not found. {}".format(lineid, subject))
                else:
                    cand_entity_names = names_map[subject]

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
    # print("skipped # questions: {}".format(skipped))
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
