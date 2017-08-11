
from args import get_args
import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from simple_qa_ner import SimpleQADataset
from model import EntityDetection
import time
import os
import glob
import numpy as np

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


train_iters = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iters = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=False)
test_iters = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
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
num_iters_in_epoch = (len(train) // args.batch_size) + 1
patience = args.patience * num_iters_in_epoch # for early stopping
iters_not_improved = 0 # this parameter is used for stopping early
early_stop = False
os.makedirs(args.save_path, exist_ok=True)



if args.test or args.dev:
    data_iters = test_iters if args.test else dev_iters
    model.eval()
    data_iters.init_epoch()
    n_data_correct = 0
    n_data_total = 0
    index2tag = np.array(labels.vocab.itos)
    index2word = np.array(questions.vocab.itos)
    for data_batch_idx, data_batch in enumerate(data_iters):
        answer = model(data_batch)
        # Get the tag from distribution, match the tag with gold label. 1 for correctness, 0 for error.
        # Sum along the sentence (dim = 0, because batch = sentence_length * batch_size)
        # If all tags are correct in one sentence, the sum should be equal to sentence length
        # Finally sum over batch_size, get the correct instance number in this batch
        n_data_correct += ((torch.max(answer, 1)[1].view(data_batch.label.size()).data == data_batch.label.data).sum(dim=0)
                            == data_batch.label.size()[0]).sum()
        n_data_total += data_batch.batch_size
        #index_tag = np.transpose(torch.max(answer, 1)[1].view(data_batch.label.size()).cpu().data.numpy())
        #tag_array = index2tag[index_tag]
        #index_question = np.transpose(data_batch.question.cpu().data.numpy())
        #question_array = index2word[index_question]
        # Print the result
        #for i in range(data_batch.batch_size):
        #    print(" ".join(question_array[i]), '\t', " ".join(tag_array[i]))

    data_acc = 100. * n_data_correct / n_data_total
    print("{} accuracy: {:10.6f}%".format("Test" if args.test else "Dev", data_acc))
    exit()

train_instance_total = len(train)
epoch = -1
#for epoch in range(args.epochs):
while True:
    epoch += 1
    if early_stop:
        print("Early stopping. Epoch: {}. Best Dev. Acc. : {}".format(epoch, best_dev_acc))
    train_iters.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iters):
        iterations += 1
        model.train()
        optimizer.zero_grad()
        answer = model(batch)
        n_correct += ((torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum(dim=0)
                       == batch.label.size()[0]).sum()
        n_total += batch.batch_size
        loss = criterion(answer, batch.label.view(-1,1)[:,0])
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()

        if iterations % args.log_every == 0:
            print("[{:6.2f} epoch] Accumulated Accuracy: {:8.4f}%".format(epoch + n_total / train_instance_total, 100. * n_correct / n_total))

        if iterations % args.dev_every == 0:
            model.eval()
            dev_iters.init_epoch()
            n_dev_correct = 0
            n_dev_total = 0
            for dev_batch_idx, dev_batch in enumerate(dev_iters):
                answer = model(dev_batch)
                n_dev_correct += ((torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum(dim=0)
                                   == dev_batch.label.size()[0]).sum()
            dev_acc = 100. * n_dev_correct / len(dev)
            print("Dev accuracy: {:10.6f}%".format(dev_acc))

            # update model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                iters_not_improved = 0
                snapshot_path = os.path.join(args.save_path, "best_model_devacc_{}_epoch_{}.pt".format(best_dev_acc, epoch + n_total / train_instance_total))
                torch.save(model, snapshot_path)
                for f in glob.glob(args.save_path + '/best_model_devacc_*'):
                    if f != snapshot_path:
                        os.remove(f)
                print("Updated and Save the best model, Accuracy on Development Set: {:8.4f}%, Epoch: {:6.2f}".format(best_dev_acc, epoch + n_total / train_instance_total))
            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break


