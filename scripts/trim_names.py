#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri, processed_text


def get_all_entity_mids(fbpath):
    mids = set()
    with open(fbpath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))

            subject = www2fb(items[0])
            mids.add(subject)

    return mids


def trim_names(fbsubsetpath, namespath, outpath):
    print("getting all entity MIDs from Freebase subset...")
    mids_to_check = get_all_entity_mids(fbsubsetpath)
    print("trimming names...")
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
            name = processed_text(clean_uri(items[2]))

            if entity in mids_to_check and name.strip() != "":
                line_to_write = "{}\t{}\t{}\n".format(entity, type, name)
                outfile.write(line_to_write)

    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trim the names file')
    parser.add_argument('-s', '--fbsubset', dest='fbsubset', action='store', required=True,
                        help='path to freebase subset file')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the names file (from CFO)')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output file path for trimmed names file')

    args = parser.parse_args()
    print("Freebase subset: {}".format(args.fbsubset))
    print("Names file from CFO: {}".format(args.names))
    print("Output trimmed names file: {}".format(args.output))

    trim_names(args.fbsubset, args.names, args.output)
    print("Trimmed the names file.")
