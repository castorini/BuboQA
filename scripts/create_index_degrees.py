#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri


def create_index_degrees(fbpath, outpath):
    print("creating the index map...")
    index = {} # indegree, outdegree
    size = 0
    with open(fbpath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))

            subject = www2fb(items[0])
            predicate = www2fb(items[1])
            object = www2fb(items[2])
            # print("{}  -   {}".format(subject, predicate))

            # increment outdegree of subject
            if subject in index.keys():
                index[subject][1] += 1
            else:
                index[subject] = [0, 1]
            # increment the count of predicate - first index, 2nd index is useless here
            if predicate in index.keys():
                index[predicate][0] += 1
            else:
                index[predicate] = [1, 0]
            # increment the indegree of object
            if object in index.keys():
                index[object][0] += 1
            else:
                index[object] = [1, 0]

            size += 1

    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    with open(outpath, 'wb') as f:
        pickle.dump(index, f)

    print("DONE")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create index for graph degree counts')
    parser.add_argument('-s', '--fbsubset', dest='fbsubset', action='store', required=True,
                        help='path to freebase subset file')
    parser.add_argument('-p', '--pickle', dest='pickle', action='store', required=True,
                        help='output file for the index pickle to be dumped')

    args = parser.parse_args()
    print("Freebase subset: {}".format(args.fbsubset))
    print("Pickle output: {}".format(args.pickle))

    create_index_degrees(args.fbsubset, args.pickle)
    print("Created the reachability index.")
