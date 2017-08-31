#!/usr/bin/python

import os
import sys
import argparse
import pickle

from fuzzywuzzy import fuzz
from util import www2fb, clean_uri

def get_index(index_path):
    print("loading index from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index

def pick_best_name(question, names_list):
    best_score = None
    best_name = None
    for name in names_list:
        score =  fuzz.ratio(name, question)
        if best_score == None or score > best_score:
            best_score = score
            best_name = name

    return best_name

def augment_dataset(datadir, index_namespath, outdir):
    names_map = get_index(index_namespath)
    skipped = 0
    allpath = os.path.join(outdir, "all.txt")
    outallfile = open(allpath, 'w')
    print("creating new datasets...")
    files = [("annotated_fb_data_train", "train"), ("annotated_fb_data_valid", "valid"), ("annotated_fb_data_test", "test")]
    for f_tuple in files:
        f = f_tuple[0]
        fname = f_tuple[1]
        fpath = os.path.join(datadir, f + ".txt")
        fpath_numbered = os.path.join(outdir, fname + ".txt")

        outfile = open(fpath_numbered, 'w')
        print("reading from {}".format(fpath))
        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0:
                    print("line: {}".format(i))

                items = line.strip().split("\t")
                if len(items) != 4:
                    print("ERROR: line - {}".format(line))
                    sys.exit(0)

                lineid = "{}-{}".format(fname, (i + 1))
                subject = www2fb(items[0])
                predicate = www2fb(items[1])
                object = www2fb(items[2])
                question = items[3]

                if subject not in names_map.keys():
                    skipped += 1
                    print("lineid {} - name not found. skipping question.".format(lineid))
                    continue

                cand_entity_names = names_map[subject]
                entity_name = pick_best_name(question, cand_entity_names)

                line_to_print = "{}\t{}\t{}\t{}\t{}\t{}".format(lineid, subject, entity_name, predicate, object, question)
                # print(line_to_print)
                outfile.write(line_to_print + "\n")
                outallfile.write(line_to_print + "\n")

        print("wrote to {}".format(fpath_numbered))
        outfile.close()

    print("wrote to {}".format(allpath))
    print("skipped # questions: {}".format(skipped))
    outallfile.close()
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment dataset with line ids, shorted names, entity names')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required = True,
                        help='path to the dataset directory - contains train, valid, test files')
    parser.add_argument('-i', '--index_names', dest='index_names', action='store', required=True,
                        help='path to the pickle for the names index')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output directory for new dataset')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Index - Names: {}".format(args.index_names))
    print("Output: {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    augment_dataset(args.dataset, args.index_names, args.output)
    print("Augmented dataset with line ids, shorted names, entity names.")
