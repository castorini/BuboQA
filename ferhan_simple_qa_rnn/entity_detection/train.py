import torch
import torch.optim as optim
import torch.nn as nn
import time
import os
import glob
import numpy as np

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

# Buckets
# train_iters, dev_iters, test_iters = data.BucketIterator.splits(
#     (train, dev, test), batch_size=args.batch_size, device=args.gpu)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=False)

# define models

config = args
config.n_embed = len(questions.vocab)
config.n_out = len(labels.vocab) # I/in entity  O/out of entity
config.n_cells = config.n_layers

if config.birnn:
    config.n_cells *= 2
print(config)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = EntityDetection(config)
    if args.word_vectors:
        model.embed.weight.data = questions.vocab.vectors
    if args.cuda:
        model.cuda()
        print("Shift model to GPU")

criterion = nn.NLLLoss() # negative log likelyhood loss function
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train the model
iterations = 0
start = time.time()
best_dev_acc = 0
best_dev_F = 0
num_iters_in_epoch = (len(train) // args.batch_size) + 1
patience = args.patience * num_iters_in_epoch # for early stopping
iters_not_improved = 0 # this parameter is used for stopping early
early_stop = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss       Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{}'.split(','))
best_snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
os.makedirs(args.save_path, exist_ok=True)
print(header)

index2tag = np.array(labels.vocab.itos)


for epoch in range(1, args.epochs+1):
    if early_stop:
        print("Early stopping. Epoch: {}, Best Dev. Acc: {}".format(epoch, best_dev_acc))
        break

    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        model.train(); optimizer.zero_grad()

        scores = model(batch)

        n_correct += ((torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum(dim=0) \
                       == batch.label.size()[0]).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(scores, batch.label.view(-1,1)[:,0])
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0],
                                                                                                iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct = 0

            gold_list = []
            pred_list = []

            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                n_dev_correct += ((torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum(dim=0) \
                                   == dev_batch.label.size()[0]).sum()
                index_tag = np.transpose(torch.max(answer, 1)[1].view(dev_batch.label.size()).cpu().data.numpy())
                gold_list.append(np.transpose(dev_batch.label.cpu().data.numpy()))
                pred_list.append(index_tag)
            P, R, F = evaluation(gold_list, pred_list, index2tag)


            dev_acc = 100. * n_dev_correct / len(dev)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          train_acc, dev_acc))
            print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * P, 100. * R, 100. * F))
            # update model
            if F > best_dev_F:
                best_dev_F = F
                iters_not_improved = 0
                snapshot_path = best_snapshot_prefix + '_devf1_{}__iter_{}_model.pt'.format(best_dev_F, iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(best_snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break

        # print progress message
        elif iterations % args.log_every == 0:
            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], n_correct/n_total*100, ' '*12))

