#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri, processed_text

def get_names_for_entities(namespath):
    print("getting names map...")
    names = {}
    with open(namespath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))
                continue
            entity = items[0]
            type = items[1]
            literal = items[2].strip()
            if literal != "":
                if names.get(entity) is None:
                    names[entity] = [(literal)]
                else:
                    names[entity].append(literal)
    return names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Created the names index")')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the trimmed names file')
    parser.add_argument('-p', '--pickle', dest='pickle', action='store', required=True,
                        help='output file path for the pickle of names index')

    args = parser.parse_args()
    print("Names file path: {}".format(args.names))
    print("Pickle output path: {}".format(args.pickle))

    index_names = get_names_for_entities(args.names)

    with open(args.pickle, 'wb') as f:
        pickle.dump(index_names, f)

    print("Created the names index.")
