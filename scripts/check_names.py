#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri

def get_index(index_path):
    print("loading index from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index


def check_names(datapath, index_namespath, outpath):
    names_map = get_index(index_namespath)
    notfound = 0
    total = 0
    outfile = open(outpath, 'w')
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
            predicate = www2fb(items[2])
            object = www2fb(items[3])
            question = items[4]

            line_to_print = "{}\t{}\t{}\t{}\t{}".format(lineid, subject, predicate, object, question)

            if subject not in names_map.keys():
                print("WARNING: name not found in map. line - {}".format(line))
                notfound += 1
                outfile.write(line_to_print + "\n")

            print(line_to_print)

    print("notfound: {}".format(notfound))
    print("found: {}".format(total-notfound))
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check which examples have entities that do not have name mappings')
    parser.add_argument('-d', '--data', dest='data', action='store', required = True,
                        help='path to the NUMBERED dataset all.txt file')
    parser.add_argument('-i', '--index_names', dest='index_names', action='store', required=True,
                        help='path to the pickle for the names index')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output file path for list of examples that do not exist')

    args = parser.parse_args()
    print("Dataset: {}".format(args.data))
    print("Index - Names: {}".format(args.index_names))
    print("Output: {}".format(args.output))

    check_names(args.data, args.index_names, args.output)
    print("Checked the names.")
