import torch
import torch.nn as nn
import time
import os
import numpy as np
from torchtext import data
from args import get_args
import random
from sq_relation_dataset import SQdataset
from relation_prediction import RelationPrediction

np.set_printoptions(threshold=np.nan)
# Set default configuration in : args.py
args = get_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

# Set up the data for training
TEXT = data.Field(lower=True)
RELATION = data.Field(sequential=False)

train, dev, test = SQdataset.splits(TEXT, RELATION, args.data_dir)
TEXT.build_vocab(train, dev, test)
RELATION.build_vocab(train, dev)

match_embedding = 0
if os.path.isfile(args.vector_cache):
    stoi, vectors, dim = torch.load(args.vector_cache)
    TEXT.vocab.vectors = torch.Tensor(len(TEXT.vocab), dim)
    for i, token in enumerate(TEXT.vocab.itos):
        wv_index = stoi.get(token, None)
        if wv_index is not None:
            TEXT.vocab.vectors[i] = vectors[wv_index]
            match_embedding += 1
        else:
            TEXT.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
else:
    print("Error: Need word embedding pt file")
    exit(1)

print("Embedding match number {} out of {}".format(match_embedding, len(TEXT.vocab)))

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

config = args
config.words_num = len(TEXT.vocab)

if args.dataset == 'RelationPrediction':
    config.rel_label = len(RELATION.vocab)
    model = RelationPrediction(config)
else:
    print("Error Dataset")
    exit()

model.embed.weight.data.copy_(TEXT.vocab.vectors)
if args.cuda:
    model.cuda()
    print("Shift model to GPU")

print(config)
print("VOCAB num",len(TEXT.vocab))
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Test instance", len(test))
print("Relation Type", len(RELATION.vocab))
print(model)

parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

criterion = nn.NLLLoss()
early_stop = False
best_dev_P = 0
iterations = 0
iters_not_improved = 0
num_dev_in_epoch = (len(train) // args.batch_size // args.dev_every) + 1
patience = args.patience * num_dev_in_epoch # for early stopping
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{},{}'.split(','))
save_path = os.path.join(args.save_path, args.relation_prediction_mode.lower())
os.makedirs(save_path, exist_ok=True)
print(header)


if args.dataset == 'RelationPrediction':
    index2tag = np.array(RELATION.vocab.itos)
else:
    print("Wrong Dataset")
    exit(1)

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_P))
        break
    epoch += 1
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    n_correct_ed, n_correct_ner , n_correct_rel = 0, 0, 0

    for batch_idx, batch in enumerate(train_iter):
        # Batch size : (Sentence Length, Batch_size)
        iterations += 1
        model.train(); optimizer.zero_grad()
        scores = model(batch)
        if args.dataset == 'RelationPrediction':
            n_correct += (torch.max(scores, 1)[1].view(batch.relation.size()).data == batch.relation.data).sum()
            loss = criterion(scores, batch.relation)
        else:
            print("Wrong Dataset")
            exit()

        n_total += batch.batch_size
        loss.backward()
        optimizer.step()

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct = 0
            n_dev_correct_rel = 0

            gold_list = []
            pred_list = []

            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)

                if args.dataset == 'RelationPrediction':
                    n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.relation.size()).data == dev_batch.relation.data).sum()
                else:
                    print("Wrong Dataset")
                    exit()

            if args.dataset == 'RelationPrediction':
                P = 1. * n_dev_correct / len(dev)
                print("{} Precision: {:10.6f}%".format("Dev", 100. * P))
            else:
                print("Wrong dataset")
                exit()

            # update model
            if args.dataset == 'RelationPrediction':
                if P > best_dev_P:
                    best_dev_P = P
                    iters_not_improved = 0
                    snapshot_path = os.path.join(save_path, args.specify_prefix + '_best_model.pt')
                    torch.save(model, snapshot_path)
                else:
                    iters_not_improved += 1
                    if iters_not_improved > patience:
                        early_stop = True
                        break
            else:
                print("Wrong dataset")
                exit()


        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                          100. * n_correct / n_total, ' ' * 12))
