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

from model import RelationClassifier
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

# ---- prepare the dataset with Torchtext -----
questions = data.Field(lower=True)
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)

# build vocab for questions
questions.build_vocab(train, dev, test)

# load word vectors if already saved or else load it from start and save it
if os.path.isfile(args.vector_cache):
    questions.vocab.vectors = torch.load(args.vector_cache)
else:
    questions.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
    os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
    torch.save(questions.vocab.vectors, args.vector_cache)

# build vocab for relations
relations.build_vocab(train, dev, test)

# create iterators
train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False)

# ---- define the model, loss, optim ------
config = args
config.n_embed = len(questions.vocab) # vocab. size / number of embeddings
config.d_out = len(relations.vocab)
config.n_cells = config.n_layers
# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2
print(config)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage,location: storage.cuda(args.gpu))
else:
    model = RelationClassifier(config)
    if args.word_vectors:
        model.embed.weight.data = questions.vocab.vectors
        if args.cuda:
            model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


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
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

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
                snapshot_path = best_snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.data[0], iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(best_snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
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

#--------TEST----------
# load the best model
print("Testing using the best model on dev set...")
best_model_path = glob.glob(best_snapshot_prefix + '*')[0]
model = torch.load(best_model_path, map_location=lambda storage,location: storage.cuda(args.gpu))
model.eval(); test_iter.init_epoch()

# calculate accuracy on test set
n_test_correct = 0
n_retrieved = 0
index2rel = np.array(relations.vocab.itos)
results_file = open(args.test_results_path, 'w')
for test_batch_idx, test_batch in enumerate(test_iter):
     scores = model(test_batch)
     n_test_correct += (torch.max(scores, 1)[1].view(test_batch.relation.size()).data == test_batch.relation.data).sum()
     # output the top results to a file
     top_k_scores, top_k_indices = torch.topk(scores, k=args.hits, dim=1, sorted=True) # shape: (batch_size, k)
     top_k_indices_array = top_k_indices.cpu().data.numpy()
     top_k_scores_array = top_k_scores.cpu().data.numpy()
     top_k_relatons_array = index2rel[top_k_indices_array] # shape: (batch_size, k)

     # write to file
     for i, (relations_row, scores_row) in enumerate(zip(top_k_relatons_array, top_k_scores_array)):
         index = (test_batch_idx * args.batch_size) + i
         example = test_batch.dataset.examples[index]  # -- problem
         results_file.write("test-{} %%%% {} %%%% {}\n".format(index+1, " ".join(example.question), example.relation))
         for rel, score in zip(relations_row, scores_row):
             results_file.write("{} %%%% {}\n".format(rel, score))
             if (rel == example.relation):
                 n_retrieved += 1
         results_file.write("-" * 60 + "\n")

retrieval_rate = 100. * n_retrieved / len(test)
print("Retrieval Rate (hits = {}): {:8.6f}".format(args.hits, retrieval_rate))
test_acc = 100. * n_test_correct / len(test)
print("Test Accuracy: {:8.6f}".format(test_acc))
results_file.close()
