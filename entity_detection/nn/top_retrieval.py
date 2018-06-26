import torch
import torch.nn as nn
import time
import os
import numpy as np
from tqdm import tqdm
from torchtext import data
from args import get_args
import random
from evaluation import evaluation
from sq_entity_dataset import SQdataset


np.set_printoptions(threshold=np.nan)
# Set default configuration in : args.py
args = get_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

TEXT = data.Field(lower=True)
ED = data.Field()

train, dev, test = SQdataset.splits(TEXT, ED, path=args.data_dir)
TEXT.build_vocab(train, dev, test)
ED.build_vocab(train, dev, test)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

print(model)

if args.dataset == 'EntityDetection':
    index2tag = np.array(ED.vocab.itos)
else:
    print("Wrong Dataset")
    exit(1)

index2word = np.array(TEXT.vocab.itos)

results_path = os.path.join(args.results_path, args.entity_detection_mode.lower())
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)


def convert(fileName, idFile, outputFile):
    fin = open(fileName)
    fid = open(idFile)
    fout = open(outputFile, "w")

    for line, line_id in tqdm(zip(fin.readlines(), fid.readlines())):
        query_list = []
        query_text = []
        line = line.strip().split('\t')
        sent = line[0].strip().split()
        pred = line[1].strip().split()
        for token, label in zip(sent, pred):
            if label == 'I':
                query_text.append(token)
            if label == 'O':
                query_text = list(filter(lambda x: x != '<pad>', query_text))
                if len(query_text) != 0:
                    query_list.append(" ".join(list(filter(lambda x:x!='<pad>', query_text))))
                    query_text = []
        query_text = list(filter(lambda x: x != '<pad>', query_text))
        if len(query_text) != 0:
            query_list.append(" ".join(list(filter(lambda x:x!='<pad>', query_text))))
            query_text = []
        if len(query_list) == 0:
            query_list.append(" ".join(list(filter(lambda x:x!='<pad>',sent))))
        fout.write(" %%%% ".join([line_id.strip()]+query_list)+"\n")


def predict(dataset_iter=test_iter, dataset=test, data_name="test"):
    print("Dataset: {}".format(data_name))
    model.eval()
    dataset_iter.init_epoch()

    n_correct = 0
    fname = "{}.txt".format(data_name)
    temp_file = 'tmp'+fname
    results_file = open(temp_file, 'w')

    gold_list = []
    pred_list = []

    for data_batch_idx, data_batch in enumerate(dataset_iter):
        scores = model(data_batch)
        if args.dataset == 'EntityDetection':
            n_correct += torch.sum(torch.sum(torch.max(scores, 1)[1].view(data_batch.ed.size()).data == data_batch.ed.data, dim=1) \
                              == data_batch.ed.size()[0]).item()
            index_tag = np.transpose(torch.max(scores, 1)[1].view(data_batch.ed.size()).cpu().data.numpy())
            tag_array = index2tag[index_tag]
            index_question = np.transpose(data_batch.text.cpu().data.numpy())
            question_array = index2word[index_question]
            gold_list.append(np.transpose(data_batch.ed.cpu().data.numpy()))
            gold_array = index2tag[np.transpose(data_batch.ed.cpu().data.numpy())]
            pred_list.append(index_tag)
            for  question, label, gold in zip(question_array, tag_array, gold_array):
                results_file.write("{}\t{}\t{}\n".format(" ".join(question), " ".join(label), " ".join(gold)))
        else:
            print("Wrong Dataset")
            exit()

    if args.dataset == 'EntityDetection':
        P, R, F = evaluation(gold_list, pred_list, index2tag, type=False)
        print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * P, 100. * R,
                                                                                     100. * F))
    else:
        print("Wrong dataset")
        exit()
    results_file.flush()
    results_file.close()
    convert(temp_file, os.path.join(args.data_dir, "lineids_{}.txt".format(data_name)), os.path.join(results_path,"query.{}".format(data_name)))
    os.remove(temp_file)


# run the model on the dev set and write the output to a file
predict(dataset_iter=dev_iter, dataset=dev, data_name="valid")

# run the model on the test set and write the output to a file
predict(dataset_iter=test_iter, dataset=test, data_name="test")





