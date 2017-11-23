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


parser = ArgumentParser(description='Logistic Regression - embeddings (GloVe) and top 300 relation word features')
parser.add_argument('--data_dir', type=str, default="../../data/lr_glove_rel_features/")
parser.add_argument('--save_path', type=str, default='./saved_checkpoints/embeddings_rel')
parser.add_argument('--results_path', type=str, default='./results/embeddings_rel/')
parser.add_argument('--trained_model', type=str, default='')
parser.add_argument('--hits', type=int, default=5, help="number of top results to output")
args = parser.parse_args()
print(args)

os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.results_path, exist_ok=True)

train_path = os.path.join(args.data_dir, "feature.train")
valid_path = os.path.join(args.data_dir, "feature.valid")
test_path = os.path.join(args.data_dir, "feature.test")
N = args.hits

X_train, y_train = [], []

if not args.trained_model:
    with open(train_path, 'r') as f:
        for line in f:
            items = line.split(" %%%% ")
            lineid = items[0].split("-")[0] + str(int(items[0].split("-")[1]))
            x = np.fromstring(items[1], sep=' ')
            X_train.append(x)
            y_train.append(items[2].strip())

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("training the model - this will take some time...")
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print("model trained, saving to pickle...")
    pickle.dump(clf, open(os.path.join(args.save_path, "lr_embeddings_rel_clf.pkl"), "wb"))

    predicted_train = clf.predict(X_train)
    accuracy_on_train = 100.0 * np.mean(predicted_train == y_train)
    print("Accuracy on training set: {}".format(accuracy_on_train))

else:
    print("loading the model from pickle...")
    clf = pickle.load(open(args.trained_model, "rb" ) )


for (data, path) in [("valid", valid_path), ("test", test_path)]:
    print("\nTesting on {} data...".format(data))
    X_test, y_test = [], []
    lineids = []
    with open(path, 'r') as f:
        for line in f:
            items = line.split(" %%%% ")
            lineid = items[0].split("-")[0] + "-" + str(int(items[0].split("-")[1]))
            lineids.append(lineid)
            x = np.fromstring(items[1], sep=' ')
            X_test.append(x)
            y_test.append(items[2].strip())

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Test the LR model
    predicted = clf.predict(X_test)
    accuracy = 100.0 * np.mean(predicted == y_test)
    print("Accuracy on {} dataset: {}".format(data, accuracy))

    probs = clf.predict_proba(X_test)
    best_n = np.argsort(probs, axis=1)[:,-N:]
    best_prob = np.sort(probs, axis=1)[:,-N:]
    best_rels = clf.classes_[best_n]

    assert(len(lineids) == len(best_rels))
    for hits in [1, 3, 5]:
        total = 0
        retrieved = 0
        for i, (top_rels, top_probs) in enumerate(zip(best_rels, best_prob)):
            total += 1
            cand_rels = list(reversed(top_rels))[:hits]
            gold = y_test[i]
            if gold in cand_rels:
                retrieved += 1

        rate = retrieved * 100.0 / total
        print("Hits: {}, Retrieved: {}, Total: {}, RetrievalRate: {}".format(hits, retrieved, total, rate))

    outfile = open(os.path.join(args.results_path, "topk-retrieval-{}-hits-{}.txt").format(data, args.hits), "w")
    for i, (top_rels, top_probs) in enumerate(zip(best_rels, best_prob)):
        gold = y_test[i]
        top_rels = reversed(top_rels)
        top_probs = reversed(top_probs)
        for rel, prob in zip(top_rels, top_probs):
            if gold == rel:
                label = 1
            else:
                label = 0
            outfile.write("{} %%%% {} %%%% {} %%%% {}\n".format(lineids[i], rel, label, prob))

print("Done!")
