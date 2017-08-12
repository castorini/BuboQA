#!/usr/bin/python

import os
import sys
import argparse

def convert_to_query_text(datadir, outdir):
    files = [("main-valid-results", "val"), ("main-test-results", "test")]
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

                items = line.strip().split(" %%%% ")
                if len(items) != 3:
                    print("ERROR: line - {}".format(line))
                    sys.exit(1)

                lineid = items[0].strip()
                tokens = items[1].strip().split()
                tags = items[2].strip().split()

                query_tokens = []
                for token, tag in zip(tokens, tags):
                    if tag == "I":
                        query_tokens.append(token)

                query_text = " ".join(query_tokens)

                line_to_print = "{} %%%% {}".format(lineid, query_text)
                # print(line_to_print)
                outfile.write(line_to_print + "\n")

        print("done with dataset: {}".format(fname))
        print("notfound: {}".format(notfound))
        print("found: {}".format(total-notfound))
        print("-" * 60)
        outfile.close()
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the gold query text after entity detection')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required = True,
                        help='path to the results directory after entity detection')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output directory for the query text')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Output: {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    convert_to_query_text(args.dataset, args.output)
    print("Converted the results after entity detection to query text.")
