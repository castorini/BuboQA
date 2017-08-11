#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri

def get_names_map(namespath):
    print("getting names map...")
    names = {}
    with open(namespath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 4:
                print("ERROR: line - {}".format(line))
            entity = clean_uri(items[0])
            type = clean_uri(items[1])
            literal = clean_uri(items[2]).lower()
            if entity not in names.keys():
                names[entity] = [literal]
            else:
                names[entity].append(literal)
    return names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Created the names index")')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the names file (from CFO)')
    parser.add_argument('-p', '--pickle', dest='pickle', action='store', required=True,
                        help='output file path for the pickle of names index')

    args = parser.parse_args()
    print("Names file path: {}".format(args.names))
    print("Pickle output path: {}".format(args.pickle))

    index_names = get_names_map(args.names)

    with open(args.pickle, 'wb') as f:
        pickle.dump(index_names, f)

    print("Created the names index.")
