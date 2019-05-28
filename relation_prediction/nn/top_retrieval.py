import torch
import os
import numpy as np
from torchtext import data
from args import get_args
import random
from sq_relation_dataset import SQdataset


np.set_printoptions(threshold=np.nan)
# Set default configuration in : args.py
args = get_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.cuda:
    args.gpu = -1
    map_location = None
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    map_location = lambda storage, location: storage.cuda(args.gpu)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

TEXT = data.Field(lower=True)
RELATION = data.Field(sequential=False)

train, dev, test = SQdataset.splits(TEXT, RELATION, args.data_dir)
TEXT.build_vocab(train, dev, test)
RELATION.build_vocab(train, dev)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

# load the model
model = torch.load(args.trained_model, map_location=map_location)

print(model)

if args.dataset == 'RelationPrediction':
    index2tag = np.array(RELATION.vocab.itos)
else:
    print("Wrong Dataset")
    exit(1)

index2word = np.array(TEXT.vocab.itos)

results_path = os.path.join(args.results_path, args.relation_prediction_mode.lower())
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

def predict(dataset_iter=test_iter, dataset=test, data_name="test"):
    print("Dataset: {}".format(data_name))
    model.eval()
    dataset_iter.init_epoch()

    n_correct = 0
    fname = "{}.txt".format(data_name)
    results_file = open(os.path.join(results_path, fname), 'w')
    n_retrieved = 0

    fid = open(os.path.join(args.data_dir,"lineids_{}.txt".format(data_name)))
    sent_id = [x.strip() for x in fid.readlines()]

    for data_batch_idx, data_batch in enumerate(dataset_iter):
        scores = model(data_batch)
        if args.dataset == 'RelationPrediction':
            n_correct += torch.sum(torch.max(scores, 1)[1].view(data_batch.relation.size()).data == data_batch.relation.data).item()
            # Get top k
            top_k_scores, top_k_indices = torch.topk(scores, k=args.hits, dim=1, sorted=True)  # shape: (batch_size, k)
            top_k_scores_array = top_k_scores.cpu().data.numpy()
            top_k_indices_array = top_k_indices.cpu().data.numpy()
            top_k_relatons_array = index2tag[top_k_indices_array]
            for i, (relations_row, scores_row) in enumerate(zip(top_k_relatons_array, top_k_scores_array)):
                index = (data_batch_idx * args.batch_size) + i
                example = data_batch.dataset.examples[index]
                for j, (rel, score) in enumerate(zip(relations_row, scores_row)):
                    if (rel == example.relation):
                        label = 1
                        n_retrieved += 1
                    else:
                        label = 0
                    results_file.write(
                        "{} %%%% {} %%%% {} %%%% {}\n".format( sent_id[index], rel, label, score))
        else:
            print("Wrong Dataset")
            exit()

    if args.dataset == 'RelationPrediction':
        P = 1. * n_correct / len(dataset)
        print("{} Precision: {:10.6f}%".format(data_name, 100. * P))
        print("no. retrieved: {} out of {}".format(n_retrieved, len(dataset)))
        retrieval_rate = 100. * n_retrieved / len(dataset)
        print("{} Retrieval Rate {:10.6f}".format(data_name, retrieval_rate))
    else:
        print("Wrong dataset")
        exit()

# run the model on the dev set and write the output to a file
predict(dataset_iter=dev_iter, dataset=dev, data_name="valid")

# run the model on the test set and write the output to a file
predict(dataset_iter=test_iter, dataset=test, data_name="test")
