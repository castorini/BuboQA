#!/usr/bin/python

import os
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

def gold_entity_detection(datadir, namespath, outdir):
    allpath = os.path.join(outdir, "all.txt")
    outallfile = open(allpath, 'w')
    names_map = get_names_for_entities(namespath)
    files = [("annotated_fb_data_train", "train"), ("annotated_fb_data_valid", "val"),
             ("annotated_fb_data_test", "test")]
    for f_tuple in files:
        f = f_tuple[0]
        fname = f_tuple[1]
        in_fpath = os.path.join(datadir, f + ".txt")
        out_fpath = os.path.join(outdir, fname + ".txt")
        notfound = 0
        total = 0
        outfile = open(out_fpath, 'w')
        print("processing dataset: {}".format(fname))
        with open(in_fpath, 'r') as f:
            for i, line in enumerate(f):
                total += 1
                if i % 1000000 == 0:
                    print("line: {}".format(i))

                items = line.strip().split("\t")
                if len(items) != 5:
                    print("ERROR: line - {}".format(line))
                    sys.exit(1)

                lineid = items[0]
                subject = www2fb(items[1])
                predicate = www2fb(items[2])
                object = www2fb(items[3])
                question = items[4]
                if subject not in names_map.keys():
                    print("WARNING: name not found in map. line - {}".format(line))
                    notfound += 1
                    continue

                name = names_map[subject][0]  # just pick the first name
                if name.strip() == "":
                    continue
                line_to_print = "{} %%%% {} %%%% {}".format(lineid, subject, name)
                # print(line_to_print)
                outfile.write(line_to_print + "\n")
                outallfile.write(line_to_print + "\n")

        print("done with dataset: {}".format(fname))
        print("notfound: {}".format(notfound))
        print("found: {}".format(total-notfound))
        print("-" * 60)
        outfile.close()

    outallfile.close()
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the gold query text after entity detection')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required = True,
                        help='path to the dataset directory - contains train, valid, test files')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the names file (from CFO)')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output directory for the gold query text after entity detection')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Names: {}".format(args.names))
    print("Output: {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    gold_entity_detection(args.dataset, args.names, args.output)
    print("Got the gold query text possible after entity detection")
