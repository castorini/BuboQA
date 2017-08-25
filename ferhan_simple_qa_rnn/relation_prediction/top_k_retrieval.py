import os
import sys
import numpy as np
import torch

from nltk.tokenize.treebank import TreebankWordTokenizer
from torchtext import data
from args import get_args
from simple_qa_relation import SimpleQaRelationDataset

# get the configuration arguments and set machine - GPU/CPU
args = get_args()
# set random seeds for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have CUDA but not using it.")
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

if not args.trained_model:
    print("ERROR: You need to provide a option 'trained_model' path to load the model.")
    sys.exit(1)

os.makedirs(args.results_path, exist_ok=True)

# ---- get the Field, Dataset, Iterator for train/dev/test sets -----
tokenizer = TreebankWordTokenizer()
def tokenize_text():
    return lambda text: tokenizer.tokenize(text)

questions = data.Field(lower=True, tokenize=tokenize_text())
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)
train_iter, dev_iter, test_iter = SimpleQaRelationDataset.iters(args, questions, relations, train, dev, test,
                                                                shuffleTrain=False)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))

index2rel = np.array(relations.vocab.itos)


def write_top_k_results(dataset_iter=train_iter, dataset=train, data_name="train"):
    print("Dataset: {}".format(data_name))
    model.eval();
    dataset_iter.init_epoch()

    n_correct = 0
    n_retrieved = 0
    fname = "topk-retrieval-{}-hits-{}.txt".format(data_name, args.hits)
    results_file = open(os.path.join(args.results_path, fname), 'w')

    for data_batch_idx, data_batch in enumerate(dataset_iter):
        scores = model(data_batch)
        n_correct += (torch.max(scores, 1)[1].view(data_batch.relation.size()).data == data_batch.relation.data).sum()
        # get the predicted top K relations
        top_k_scores, top_k_indices = torch.topk(scores, k=args.hits, dim=1, sorted=True)  # shape: (batch_size, k)
        top_k_indices_array = top_k_indices.cpu().data.numpy()
        top_k_scores_array = top_k_scores.cpu().data.numpy()
        top_k_relatons_array = index2rel[top_k_indices_array]  # shape: (batch_size, k)

        # write to file
        for i, (relations_row, scores_row) in enumerate(zip(top_k_relatons_array, top_k_scores_array)):
            index = (data_batch_idx * args.batch_size) + i
            example = data_batch.dataset.examples[index]
            # correct_relation = index2rel[test_batch.relation.data[i]]
            # results_file.write("{}-{} %%%% {} %%%% {}\n".format(data_name, index+1, " ".join(example.question), example.relation))
            found = (False, -1)
            for i, (rel, score) in enumerate(zip(relations_row, scores_row)):
                # results_file.write("{} %%%% {}\n".format(rel, score))
                if (rel == example.relation):
                    label = 1
                    n_retrieved += 1
                    found = (True, i)
                else:
                    label = 0
                results_file.write(
                    "{}-{} %%%% {} %%%% {} %%%% {}\n".format(data_name, index + 1, rel, label, score))

                # if found[0] == True:
                #     results_file.write("FOUND at index: {}\n".format(found[1]))
                # else:
                #     results_file.write("NOT found\n")
                # results_file.write("-" * 60 + "\n")

    print("no. retrieved: {} out of {}".format(n_retrieved, len(dataset)))
    retrieval_rate = 100. * n_retrieved / len(dataset)
    print("{} retrieval rate (hits = {}): {:8.6f}%".format(data_name, args.hits, retrieval_rate))
    print("no. correct: {} out of {}".format(n_correct, len(dataset)))
    accuracy = 100. * n_correct / len(dataset)
    print("{} accuracy: {:8.6f}%".format(data_name, accuracy))
    print("-" * 80)
    results_file.close()


# write out top K retrieval results for train/dev/test
write_top_k_results(dataset_iter=train_iter, dataset=train, data_name="train")
write_top_k_results(dataset_iter=dev_iter, dataset=dev, data_name="valid")
write_top_k_results(dataset_iter=test_iter, dataset=test, data_name="test")
