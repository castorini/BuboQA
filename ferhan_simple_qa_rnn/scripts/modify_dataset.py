#!/usr/bin/python

import os
import sys
import argparse

from util import www2fb, clean_uri


def create_new_dataset(datadir, outdir):
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

                line_to_print = "{}\t{}\t{}\t{}\t{}".format(lineid, subject, predicate, object, question)
                # print(line_to_print)
                outfile.write(line_to_print + "\n")
                outallfile.write(line_to_print + "\n")

        print("wrote to {}".format(fpath_numbered))
        outfile.close()

    print("wrote to {}".format(allpath))
    outallfile.close()
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify dataset with line ids and shorted entity, relation names')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required = True,
                        help='path to the dataset directory - contains train, valid, test files')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output directory for new dataset')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Output: {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    create_new_dataset(args.dataset, args.output)
    print("Modified the dataset with line ids and shorted entity, relation names.")
