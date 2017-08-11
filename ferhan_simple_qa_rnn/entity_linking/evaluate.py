#!/usr/bin/python

import os
import sys
import argparse

from util import www2fb, clean_uri

def get_predictions(datapath):
    lineids = []
    id2pred = {}
    with open(datapath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            lineid = items[0].strip()
            ent_mid = items[1].strip()
            rel = items[2].strip()
            # print("{}   -   {}".format(lineid, rel))
            lineids.append(lineid)
            id2rel[lineid] = rel
    return lineids, id2rel

def get_gold_labels(datapath):
    lineids = []
    id2label = {}
    with open(datapath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 5:
                print("ERROR: line - {}".format(line))
                sys.exit(1)

            lineid = items[0]
            subject = www2fb(items[1])
            predicate = www2fb(items[2])
            lineids.append(lineid)
            id2label[lineid] = (subject, predicate)
    return lineids, id2label


def evaluate(goldpath, predpath):
    allpath = os.path.join(outdir, "all.txt")
    outallfile = open(allpath, 'w')
    names_map = names_map = get_index(index_namespath)
    # files = [("annotated_fb_data_train", "train"), ("annotated_fb_data_valid", "val"), ("annotated_fb_data_test", "test")]
    files = [("train", "train"), ("val", "val"), ("test", "test")]
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

                name = names_map[subject]
                if name.strip() == "":
                    print("WARNING: name stripped empty. line - {}".format(line))
                    notfound += 1
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
    parser.add_argument('-g', '--gold', dest='gold', action='store', required = True,
                        help='path to the NUMBERED test dataset file - test.txt')
    parser.add_argument('-p', '--prediction', dest='prediction', action='store', required=True,
                        help='path to the entity linking predictions file')

    args = parser.parse_args()
    print("Gold Test Dataset: {}".format(args.gold))
    print("Predictions on Test Dataset: {}".format(args.prediction))

    evaluate(args.gold, args.prediction)
    print("Done evaluating.")
