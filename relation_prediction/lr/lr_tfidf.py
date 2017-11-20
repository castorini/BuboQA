import os
import numpy as np
import pandas as pd
import pickle

from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics


parser = ArgumentParser(description='Logistic Regression - tfidf on unigrams and bigrams')
parser.add_argument('--data_dir', type=str, default="../../data/processed_simplequestions_dataset/")
parser.add_argument('--save_path', type=str, default='./saved_checkpoints/tfidf/')
parser.add_argument('--results_path', type=str, default='./results/tfidf/')
parser.add_argument('--trained_model', type=str, default='')
parser.add_argument('--hits', type=int, default=5, help="number of top results to output")
args = parser.parse_args()
print(args)

os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.results_path, exist_ok=True)

data_dir = args.data_dir

if not args.trained_model:
    train_path = os.path.join(data_dir, "train.txt")
    train_df = pd.read_table(train_path, header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
    # Logistic regression with tf-idf on unigrams and bigrams
    clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression())
                   ])
    clf.fit(train_df['question'], train_df['relation'])

    pickle.dump(clf, open(os.path.join(args.save_path, "lr_tfidf_clf.pkl"), "wb"))

else:
    clf = pickle.load(open(args.trained_model, "rb" ) )


valid_path = os.path.join(data_dir, "valid.txt")
test_path = os.path.join(data_dir, "test.txt")

valid_df = pd.read_table(valid_path, header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test_df = pd.read_table(test_path, header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
N = args.hits

for (data, df) in [("valid", valid_df), ("test", test_df)]:
    print("\nTesting on {} data...".format(data))
    predicted = clf.predict(df['question'])
    accuracy = 100.0 * np.mean(predicted == df['relation'])
    print("Accuracy on {} set: {}".format(data, accuracy))

    probs = clf.predict_proba(df['question'])
    best_n = np.argsort(probs, axis=1)[:,-N:]
    best_prob = np.sort(probs, axis=1)[:,-N:]
    best_rels = clf.classes_[best_n]

    for hits in [1, 3, 5]:
        total = 0
        retrieved = 0
        for i, (top_rels, top_probs) in enumerate(zip(best_rels, best_prob)):
            total += 1
            cand_rels = list(reversed(top_rels))[:hits]
            gold = df['relation'][i]
            if gold in cand_rels:
                retrieved += 1

        rate = retrieved * 100.0 / total
        print("Hits: {}, Retrieved: {}, Total: {}, RetrievalRate: {}".format(hits, retrieved, total, rate))


    lineids = df['lineid'].tolist()
    outfile = open(os.path.join(args.results_path, "topk-retrieval-{}-hits-{}.txt").format(data, args.hits), "w")
    for i, (top_rels, top_probs) in enumerate(zip(best_rels, best_prob)):
        gold = df['relation'][i]
        top_rels = reversed(top_rels)
        top_probs = reversed(top_probs)
        for rel, prob in zip(top_rels, top_probs):
            if gold == rel:
                label = 1
            else:
                label = 0
            outfile.write("{} %%%% {} %%%% {} %%%% {}\n".format(lineids[i], rel, label, prob))

print("Done!")
