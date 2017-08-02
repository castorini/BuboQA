#!/usr/bin/python

import os
import sys
import argparse

from util import www2fb, clean_uri

def entity_linking(index_entpath, index_reachpath, ent_resultpath, rel_resultpath, outpath):
    outfile = open(os.path.join(outpath, "linking-results.txt"), 'w')

    # index_ent = get_entity_index(index_entpath)
    # index_reach = get_reachability_index(index_reachpath)
    # query_texts = get_query_text(ent_resultpath)
    # relations = get_relations(rel_resultpath)


    outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do entity linking')
    parser.add_argument('--index_ent', dest='index_ent', action='store', required = True,
                        help='path to the pickle for the inverted entity index')
    parser.add_argument('--index_reach', dest='index_reach', action='store', required=True,
                        help='path to the pickle for the graph reachability index')
    parser.add_argument('--ent_result', dest='ent_result', action='store', required=True,
                        help='file path to the entity detection results with the query texts')
    parser.add_argument('--rel_result', dest='rel_result', action='store', required=True,
                        help='file path to the relation prediction results')
    parser.add_argument('--output', dest='output', action='store', required=True,
                        help='directory path to the output of entity linking')


    args = parser.parse_args()
    print("Index - Entity: {}".format(args.index_ent))
    print("Index - Reachability: {}".format(args.index_reach))
    print("Entity Detection Results: {}".format(args.ent_result))
    print("Relation Prediction Results: {}".format(args.rel_result))
    print("Output: {}".format(args.output))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    entity_linking(args.index_ent, args.index_reach, args.ent_result, args.rel_result, args.output)
    print("Entity Linking done.")
