#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri


def get_all_entity_mids(datapath):
    mids = set()
    with open(datapath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 5:
                print("ERROR: line - {}".format(line))
                sys.exit(0)

            lineid = items[0]
            subject = www2fb(items[1])
            mids.add(subject)
    return mids


def trim_names(datapath, namespath, outpath):
    mids_to_check = get_all_entity_mids(datapath=datapath)
    outfile = open(outpath, 'w')
    with open(namespath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 4:
                print("ERROR: line - {}".format(line))
            entity = www2fb(clean_uri(items[0]))
            type = clean_uri(items[1])

            if entity in mids_to_check:
                outfile.write(line)

    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trim the names file')
    parser.add_argument('-d', '--data', dest='data', action='store', required=True,
                        help='path to the NUMBERED dataset all.txt file')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the names file (from CFO)')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output file path for trimmed names file')

    args = parser.parse_args()
    print("Dataset: {}".format(args.data))
    print("Names file from CFO: {}".format(args.names))
    print("Output trimmed names file: {}".format(args.output))

    trim_names(args.data, args.names, args.output)
    print("Trimmed the names file.")
