#!/usr/bin/python

import sys
import argparse
import pickle

from nltk.tokenize.treebank import TreebankWordTokenizer
from util import www2fb, clean_uri, strip_accents

tokenizer = TreebankWordTokenizer()

def get_all_ngrams(tokens):
    all_ngrams = set()
    max_n = min(len(tokens), 3)
    for n in range(1, max_n+1):
        ngrams = find_ngrams(tokens, n)
        all_ngrams = all_ngrams | ngrams
    return all_ngrams

def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)


def get_name_ngrams(entity_name):
    entity_name = entity_name.lower() # lowercase the name
    name_tokens = tokenizer.tokenize(entity_name)
    name_ngrams = get_all_ngrams(name_tokens)

    return name_ngrams


def create_inverted_index_entity(namespath, outpath):
    print("creating the index map...")
    index = {}
    size = 0
    with open(namespath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 4:
                print("ERROR: line - {}".format(line))

            entity_mid = clean_uri(items[0])
            entity_type = clean_uri(items[1])
            entity_name = clean_uri(items[2])

            name_ngrams = get_name_ngrams(entity_name)

            for ngram_tuple in name_ngrams:
                size += 1
                ngram = " ".join(ngram_tuple)
                ngram = strip_accents(ngram)
                # print(ngram)
                if ngram in index.keys():
                    index[ngram].add(entity_mid)
                else:
                    index[ngram] = set([entity_mid])


    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    print("dumping to pickle...")
    with open(outpath, 'wb') as f:
        pickle.dump(index, f)

    print("DONE")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create inverted index for entity')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the trimmed names file')
    parser.add_argument('-p', '--pickle', dest='pickle', action='store', required=True,
                        help='output file for the index pickle to be dumped')

    args = parser.parse_args()
    print("Names: {}".format(args.names))
    print("Pickle output: {}".format(args.pickle))

    create_inverted_index_entity(args.names, args.pickle)
    print("Created the entity index.")
