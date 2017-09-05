import os
import sys
import numpy as np
import torch

from torchtext import data
from args import get_args
from simple_qa_ner import SimpleQADataset
from model import EntityDetection

from evaluation import evaluation

# please set the configuration in the file : args.py
args = get_args()
# set the random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")


if not args.trained_model:
    print("ERROR: You need to provide a option 'trained_model' path to load the model.")
    sys.exit(1)

# load data with torchtext
questions = data.Field(lower=True, sequential=True)
labels = data.Field(sequential=True)

train, dev, test = SimpleQADataset.splits(questions, labels)

# build vocab for questions
questions.build_vocab(train, dev, test) # Test dataset can not be used here for constructing the vocab
# build vocab for tags
labels.build_vocab(train, dev, test)

if os.path.isfile(args.vector_cache):
    questions.vocab.vectors = torch.load(args.vector_cache)
else:
    questions.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
    os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
    torch.save(questions.vocab.vectors, args.vector_cache)

# get iterators
train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=False)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=False)


# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

index2tag = np.array(labels.vocab.itos)
index2word = np.array(questions.vocab.itos)

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

    gold_list = []
    pred_list = []

    for data_batch_idx, data_batch in enumerate(dataset_iter):
        scores = model(data_batch)
        n_correct += ((torch.max(scores, 1)[1].view(data_batch.label.size()).data ==
                            data_batch.label.data).sum(dim=0) == data_batch.label.size()[0]).sum()

        index_tag = np.transpose(torch.max(scores, 1)[1].view(data_batch.label.size()).cpu().data.numpy())
        tag_array = index2tag[index_tag]
        index_question = np.transpose(data_batch.question.cpu().data.numpy())
        question_array = index2word[index_question]

        # print and write the result
        for i in range(data_batch.batch_size):
            line_to_print = "{}-{} %%%% {} %%%% {}".format(data_name, linenum, " ".join(question_array[i]), " ".join(tag_array[i]))
            # print(line_to_print)
            results_file.write(line_to_print + "\n")
            linenum += 1
        gold_list.append(np.transpose(data_batch.label.cpu().data.numpy()))
        pred_list.append(index_tag)

    #print("no. correct: {} out of {}".format(n_correct, len(dataset)))
    #accuracy = 100. * n_correct / len(dataset)
    #print("{} accuracy: {:8.6f}%".format(data_name, accuracy))
    #print("-" * 80)
    P, R, F = evaluation(gold_list, pred_list, index2tag)
    print("Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format(100. * P, 100. * R, 100. * F))
    results_file.close()


# run the model on the dev set and write the output to a file
predict(dataset_iter=dev_iter, dataset=dev, data_name="valid")

# run the model on the test set and write the output to a file
predict(dataset_iter=test_iter, dataset=test, data_name="test")
