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

# ---- get the Field, Dataset, Iterator for train/dev/test sets -----
questions = data.Field(lower=True, tokenize="moses")
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)
train_iter, dev_iter, test_iter = SimpleQaRelationDataset.iters(args, questions, relations, train, dev, test, shuffleTrain=False)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

index2rel = np.array(relations.vocab.itos)

if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)

def predict(dataset_iter=test_iter, dataset=test, data_name="test"):
    print("Dataset: {}".format(data_name))
    model.eval();
    dataset_iter.init_epoch()

    n_correct = 0
    linenum = 1
    fname = "main-{}-results.txt".format(data_name)
    results_file = open(os.path.join(args.results_path, fname), 'w')

    for test_batch_idx, test_batch in enumerate(test_iter):
        scores = model(test_batch)
        n_correct += (torch.max(scores, 1)[1].view(test_batch.relation.size()).data == test_batch.relation.data).sum()
        # get the predicted relations
        top_scores, top_indices = torch.max(scores, dim=1) # shape: (batch_size, 1)
        top_indices_array = top_indices.cpu().data.numpy().reshape(-1)
        top_scores_array = top_scores.cpu().data.numpy().reshape(-1)
        top_relatons_array = index2rel[top_indices_array] # shape: vector of dim: batch_size
        # write to file
        for i in range(test_batch.batch_size):
            line_to_print = "{}-{} %%%% {} %%%% {}".format(data_name, linenum, top_relatons_array[i], top_scores_array[i])
            results_file.write(line_to_print + "\n")
            linenum += 1

    accuracy = 100. * n_correct / len(test)
    print("{} accuracy: {:8.6f}".format(data_name, accuracy))
    results_file.close()


# run the model on the dev set and write the output to a file
predict(dataset_iter=dev_iter, dataset=dev, data_name="valid")

# run the model on the test set and write the output to a file
predict(dataset_iter=test_iter, dataset=test, data_name="test")