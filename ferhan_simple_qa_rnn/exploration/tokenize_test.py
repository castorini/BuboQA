import sys
import nltk
from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer

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

fname = "../data/SimpleQuestions_v2_modified/all.txt"

with open(fname, 'r') as f:
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print("line: {}".format(i))

        items = line.strip().split("\t")
        if len(items) != 5:
            print("ERROR: line - {}".format(line))
            sys.exit(0)

        lineid = items[0]
        subject = www2fb(items[1])
        predicate = www2fb(items[2])
        question = items[4].lower()

        tokenizer = TreebankWordTokenizer()
        # tokenizer = MosesTokenizer()
        tokens = tokenizer.tokenize(question)
        print("{} - {}".format(lineid, tokens))