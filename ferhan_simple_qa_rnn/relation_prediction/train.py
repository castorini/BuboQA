import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torchtext import data
from nltk.tokenize.treebank import TreebankWordTokenizer
from model import RelationClassifier
from args import get_args
from simple_qa_relation import SimpleQaRelationDataset

def save_checkpoint(state, snapshot_prefix, dev_train, acc, loss, iterations):
    print("saving checkpoint")
    snapshot_path = (snapshot_prefix + '_' + dev_train + 'acc_{:.4f}_'.format(acc) + 
                    dev_train + 'loss_{:.6f}_iter_{}_model.pt'.format(loss, iterations))
    torch.save(state, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)
    print("finished saving checkpoint")

def load_checkpoint(model, optimizer, resume):
    print("loading checkpoint")
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage,location: storage.cuda(args.gpu))
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))

# get the configuration arguments and set machine - GPU/CPU
args = get_args()
# set random seeds for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
else:
    args.gpu = 1
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have CUDA but not using it.")
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

# ---- get the Field, Dataset, Iterator for train/dev/test sets -----
tokenizer = TreebankWordTokenizer()
def tokenize_text():
    return lambda text: tokenizer.tokenize(text)

questions = data.Field(lower=True, tokenize=tokenize_text())
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)
train_iter, dev_iter, test_iter = SimpleQaRelationDataset.iters(args, questions, relations, train, dev, test)

# ---- define the model, loss, optim ------
config = args
config.n_embed = len(questions.vocab) # vocab. size / number of embeddings
config.d_out = len(relations.vocab)
config.n_cells = config.n_layers
# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2
print(config)

model = RelationClassifier(config)
print("relation classifier initialized")
if args.word_vectors:
    model.embed.weight.data = questions.vocab.vectors
    print("word vectors copied")
    if args.cuda:
        model.cuda()
        print("cuda called")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_weight_decay)
print("optimizer initialized")

if args.resume_snapshot:
    load_checkpoint(model, optimizer, args.resume_snapshot)

# ---- train the model ------
iterations = 0
start = time.time()
best_dev_acc = -1
num_iters_in_epoch = (len(train) // args.batch_size) + 1
patience = args.patience * num_iters_in_epoch # for early stopping
early_stop = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
best_snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
os.makedirs(args.save_path, exist_ok=True)
print(header)

print("starting training")
for epoch in range(1, args.epochs+1):
    if early_stop:
        print("Early stopping. Epoch: {}, Best Dev. Acc: {}".format(epoch, best_dev_acc))
        break

    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1

        # switch model to training mode, clear gradient accumulators
        model.train(); optimizer.zero_grad()

        # forward pass
        scores = model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(scores, 1)[1].view(batch.relation.size()).data == batch.relation.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels & backpropagate to compute gradients
        loss = criterion(scores, batch.relation)
        loss.backward()

        # clip the gradients (prevent exploding gradients) and update the weights
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')

            save_checkpoint({'epoch': epoch + 1,
			     'state_dict': model.state_dict(),
			     'optimizer' : optimizer.state_dict()},
                             snapshot_prefix, '', train_acc, loss.data[0], iterations)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct = 0
            dev_losses = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                 scores = model(dev_batch)
                 n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.relation.size()).data == dev_batch.relation.data).sum()
                 dev_loss = criterion(scores, dev_batch.relation)
                 dev_losses.append(dev_loss.data[0])
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], sum(dev_losses)/len(dev_losses), train_acc, dev_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:
                iters_not_improved = 0
                # found a model with better validation set accuracy
                best_dev_acc = dev_acc
                # save model, delete previous 'best_snapshot' files
                save_checkpoint({'epoch': epoch + 1,
	            	     'state_dict': model.state_dict(),
	            	     'optimizer' : optimizer.state_dict()},
			     best_snapshot_prefix, "dev" , dev_acc, dev_loss.data[0], iterations)
            else:
                iters_not_improved += 1
                if iters_not_improved >= patience:
                    early_stop = True
                    break

        elif iterations % args.log_every == 0:
            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

