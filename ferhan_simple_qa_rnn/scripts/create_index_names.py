#!/usr/bin/python

import sys
import argparse
import pickle

from fuzzywuzzy import fuzz
from util import www2fb, clean_uri

def get_names_for_entities(namespath):
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


def pick_best_name(question, names_list):
    best_score = None
    best_name = None
    for name in names_list:
        score =  fuzz.ratio(name, question)
        if best_score == None or score > best_score:
            best_score = score
            best_name = name

    return best_name


def get_best_name_map(namespath, datapath):
    names_list_map = get_names_for_entities(namespath)
    names_best_map = {}
    notfound = 0
    total = 0
    with open(datapath, 'r') as f:
        for i, line in enumerate(f):
            total += 1
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 5:
                print("ERROR: line - {}".format(line))
                sys.exit(0)

            lineid = items[0]
            subject = www2fb(items[1])
            question = items[4]

            if subject not in names_list_map.keys():
                print("WARNING: name not found in map. line - {}".format(line))
                notfound += 1
                continue

            best_name = pick_best_name(question, names_list_map[subject])
            names_best_map[subject] = best_name

    print("notfound: {}".format(notfound))
    print("found: {}".format(total-notfound))
    return names_best_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Created the names index")')
    parser.add_argument('-d', '--data', dest='data', action='store', required=True,
                        help='path to the NUMBERED dataset all.txt file')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the trimmed names file')
    parser.add_argument('-p', '--pickle', dest='pickle', action='store', required=True,
                        help='output file path for the pickle of names index')

    args = parser.parse_args()
    print("Dataset: {}".format(args.data))
    print("Names file path: {}".format(args.names))
    print("Pickle output path: {}".format(args.pickle))

    index_names = get_best_name_map(args.names, args.data)

    with open(args.pickle, 'wb') as f:
        pickle.dump(index_names, f)

    print("Created the names index.")
