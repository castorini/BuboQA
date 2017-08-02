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

os.makedirs(args.results_path)

# ---- get the Field, Dataset, Iterator for train/dev/test sets -----
questions = data.Field(lower=True)
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)
train_iter, dev_iter, test_iter = SimpleQaRelationDataset.iters(args, questions, relations, train, dev, test, shuffleTrain=False)

# load the model
model = torch.load(args.resume_snapshot, map_location=lambda storage,location: storage.cuda(args.gpu))

# run the model on the test set and write the output to a file
# calculate accuracy on test set
n_test_correct = 0
test_linenum = 1
index2rel = np.array(relations.vocab.itos)
results_file = open(args.test_results_path, 'w')
for test_batch_idx, test_batch in enumerate(test_iter):
    scores = model(test_batch)
    n_test_correct += (torch.max(scores, 1)[1].view(test_batch.relation.size()).data == test_batch.relation.data).sum()
    # output the top results to a file
    top_scores, top_indices = torch.topk(scores, dim=1) # shape: (batch_size, 1)
    top_indices_array = top_indices.cpu().data.numpy()
    top_scores_array = top_scores.cpu().data.numpy()
    top_relatons_array = index2rel[top_indices_array] # shape: (batch_size, 1)
    # write to file
    for i in range(len(test_batch)):
        line_to_print = "{}-{} %%%% {} %%%% {}".format("test", test_linenum, top_relatons_array[i], top_scores_array[i])
        results_file.write(line_to_print + "\n")
        test_linenum += 1


test_acc = 100. * n_test_correct / len(test)
print("Test Accuracy: {:8.6f}".format(test_acc))
results_file.close()