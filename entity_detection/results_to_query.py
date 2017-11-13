#!/usr/bin/python

import os
import sys
import argparse

# python results_to_query.py -d ../data/SimpleQuestions_v2_modified/all.txt -r results -o query-text/

def get_questions(datapath):
    print("getting questions...")
    id2question = {}
    with open(datapath, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            lineid = items[0].strip()
            sub = items[1].strip()
            pred = items[2].strip()
            obj = items[3].strip()
            question = items[4].strip()
            #print("{}   -   {}".format(lineid, question))
            id2question[lineid] = question
    return id2question

def get_span(label):
    span = []
    st = 0
    en = 0
    flag = False
    for k, l in enumerate(label):
        if l == 'I' and flag == False:
            st = k
            flag = True
        if l != 'I' and flag == True:
            flag = False
            en = k
            span.append((st, en))
            st = 0
            en = 0
    if st != 0 and en == 0:
        en = k
        span.append((st, en))
    return span

def convert_to_query_text(datapath, resultdir, outdir):
    id2question = get_questions(datapath)
    files = [("main-valid-results", "val"), ("main-test-results", "test")]
    for f_tuple in files:
        f = f_tuple[0]
        fname = f_tuple[1]
        in_fpath = os.path.join(resultdir, f + ".txt")
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
                spans = get_span(tags)
                for span in spans:
                    query_tokens.append(" ".join(tokens[span[0]:span[1]]))

                outfile.write(lineid)
                if len(query_tokens) == 0:
                    query_text = id2question[lineid]
                    query_tokens.append(query_text)
                # if no query text found, use the entire question as query
                for token in query_tokens:
                    line_to_print = " %%%% {}".format(token)
                    # print(line_to_print)
                    outfile.write(line_to_print)
                outfile.write("\n")

        print("done with dataset: {}".format(fname))
        print("notfound: {}".format(notfound))
        print("found: {}".format(total-notfound))
        print("-" * 60)
        outfile.close()
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the gold query text after entity detection')
    parser.add_argument('-r', '--result', dest='result', action='store', required = True,
                        help='path to the results directory after entity detection')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required=True,
                        help='path to the NUMBERED dataset all.txt file')
    parser.add_argument('-o', '--output', dest='output', action='store', required=True,
                        help='output directory for the query text')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Result: {}".format(args.result))
    print("Output: {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    convert_to_query_text(args.dataset, args.result, args.output)
    print("Converted the results after entity detection to query text.")
