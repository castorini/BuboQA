#!/usr/bin/python

import sys
import argparse

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
            literal = clean_uri(items[2])
            if entity not in names.keys():
                names[entity] = [literal]
            else:
                names[entity].append(literal)
    return names

def check_names(datapath, namespath, outpath):
    names_map = get_names_for_entities(namespath)
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

            line_to_print = "{}\t{}\t{}\t{}\t{}".format(lineid, subject, predicate, object, question);

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
                        help='path to the NUMBERED dataset all-data.txt file')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the names file (from CFO)')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output file path for list of examples that do not exist')

    args = parser.parse_args()
    print("Dataset: {}".format(args.data))
    print("Names: {}".format(args.names))
    print("Output: {}".format(args.output))

    check_names(args.data, args.names, args.output)
    print("Checked the names.")
