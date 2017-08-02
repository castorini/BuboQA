import os
import sys
import numpy as np
import torch

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
questions = data.Field(lower=True, tokenize="moses")
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)
train_iter, dev_iter, test_iter = SimpleQaRelationDataset.iters(args, questions, relations, train, dev, test, shuffleTrain=False)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

def write_top_results(dataset_iter=train_iter, dataset=train, data_name="train"):
    print("Dataset: {}".format(data_name))
    model.eval(); dataset_iter.init_epoch()

    # calculate accuracy on test set
    n_test_correct = 0
    n_retrieved = 0
    index2rel = np.array(relations.vocab.itos)
    fname = "{}-hits-{}.txt".format(data_name, args.hits)
    results_file = open(os.path.join(args.results_path, "topk-retrieval-" + fname), 'w')
    for data_batch_idx, data_batch in enumerate(dataset_iter):
         scores = model(data_batch)
         n_test_correct += (torch.max(scores, 1)[1].view(data_batch.relation.size()).data == data_batch.relation.data).sum()
         # output the top results to a file
         top_k_scores, top_k_indices = torch.topk(scores, k=args.hits, dim=1, sorted=True) # shape: (batch_size, k)
         top_k_indices_array = top_k_indices.cpu().data.numpy()
         top_k_scores_array = top_k_scores.cpu().data.numpy()
         top_k_relatons_array = index2rel[top_k_indices_array] # shape: (batch_size, k)

         # write to file
         for i, (relations_row, scores_row) in enumerate(zip(top_k_relatons_array, top_k_scores_array)):
             index = (data_batch_idx * args.batch_size) + i
             example = data_batch.dataset.examples[index]
             # correct_relation = index2rel[test_batch.relation.data[i]]
             results_file.write("{}-{} %%%% {} %%%% {}\n".format(data_name, index+1, " ".join(example.question), example.relation))
             found = (False, -1)
             for i, (rel, score) in enumerate(zip(relations_row, scores_row)):
                 results_file.write("{} %%%% {}\n".format(rel, score))
                 if (rel == example.relation):
                     n_retrieved += 1
                     found = (True, i)
             if found[0] == True:
                 results_file.write("FOUND at index: {}\n".format(found[1]))
             else:
                 results_file.write("NOT found\n")
             results_file.write("-" * 60 + "\n")

    retrieval_rate = 100. * n_retrieved / len(dataset)
    print("Retrieval Rate (hits = {}): {:8.6f}".format(args.hits, retrieval_rate))
    test_acc = 100. * n_test_correct / len(dataset)
    print("Accuracy: {:8.6f}".format(test_acc))
    results_file.close()

write_top_results(dataset_iter=train_iter, dataset=train, data_name="train")
write_top_results(dataset_iter=dev_iter, dataset=dev, data_name="valid")
write_top_results(dataset_iter=test_iter, dataset=test, data_name="test")
