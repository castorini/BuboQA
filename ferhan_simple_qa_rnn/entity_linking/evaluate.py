#!/usr/bin/python

import os
import sys
import argparse


def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def clean_uri(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return clean_uri(uri[1:-1])
    elif uri.startswith("\"") and uri.endswith("\""):
        return clean_uri(uri[1:-1])
    return uri

def get_predictions(datapath):
    lineids = []
    id2pred = {}
    with open(datapath, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            lineid = items[0].strip()
            mid = items[1].strip()
            rel = items[2].strip()
            # print("{}  -  {}  -  {}".format(lineid, mid, rel))
            lineids.append(lineid)
            id2pred[lineid] = (mid, rel)
    return lineids, id2pred

def get_gold_labels(datapath):
    lineids = []
    id2label = {}
    with open(datapath, 'r') as f:
        for i, line in enumerate(f):
            items = line.strip().split("\t")
            if len(items) != 5:
                print("ERROR: line - {}".format(line))
                sys.exit(1)

            lineid = items[0]
            subject = www2fb(items[1])
            predicate = www2fb(items[2])
            # print("{}  -  {}  -  {}".format(lineid, subject, predicate))
            lineids.append(lineid)
            id2label[lineid] = (subject, predicate)
    return lineids, id2label


def evaluate(goldpath, predpath):
    gold_lineids, gold_id2labels = get_gold_labels(goldpath)
    pred_lineids, pred_id2preds = get_predictions(predpath)

    not_found = 0
    found = 0

    correct = 0
    mid_wrong = 0
    rel_wrong = 0
    both_wrong = 0

    for lineid in gold_lineids:
        if lineid in pred_lineids:
            found += 1
            pred_mid, pred_rel = pred_id2preds[lineid]
            gold_mid, gold_rel = gold_id2labels[lineid]
            if pred_mid != gold_mid and pred_rel != gold_rel:
                both_wrong += 1
            elif pred_mid != gold_mid:
                mid_wrong += 1
            elif pred_rel != gold_rel:
                rel_wrong += 1
            else:
                correct += 1
        else:
            not_found += 1
            continue

    total = found + not_found
    all_wrong = both_wrong + mid_wrong + rel_wrong
    accuracy = 100.0 * (correct / total)

    print("\ntotal: {}".format(total))
    print("found: {}".format(found))
    print("not found: {}".format(not_found))

    print("\nboth wrong: {}".format(both_wrong))
    print("only mid wrong: {}".format(mid_wrong))
    print("only rel wrong: {}".format(rel_wrong))

    print("all wrong: {}".format(all_wrong))
    print("all wrong (including not found): {}".format(all_wrong + not_found))
    print("correct: {}".format(correct))
    print("\nAccuracy: {}%".format(accuracy))


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
