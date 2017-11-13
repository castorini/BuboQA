import os
import sys
import numpy as np
import torch

from torchtext import data
from nltk.tokenize.treebank import TreebankWordTokenizer
from args import get_args
from simple_qa_relation import SimpleQaRelationDataset
from model import RelationClassifier

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
tokenizer = TreebankWordTokenizer()
def tokenize_text():
    return lambda text: tokenizer.tokenize(text)

questions = data.Field(lower=True, tokenize=tokenize_text())
relations = data.Field(sequential=False)

train, dev, test = SimpleQaRelationDataset.splits(questions, relations)
train_iter, dev_iter, test_iter = SimpleQaRelationDataset.iters(args, questions, relations, train, dev, test, shuffleTrain=False)

# load the model

config = args
config.n_embed = len(questions.vocab) # vocab. size / number of embeddings
config.d_out = len(relations.vocab)
config.n_cells = config.n_layers
# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2
print(config)

model = RelationClassifier(config)
if args.word_vectors:
    model.embed.weight.data = questions.vocab.vectors
    if args.cuda:
        model.cuda()
checkpoint = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))
model.load_state_dict(checkpoint)

index2rel = np.array(relations.vocab.itos)

input_sent = "where was sasha vujacic born?"

class ins(object):
    def __init__(self, question):
      self.question = question


def get_relation(input_sent):
    sent = tokenizer.tokenize(input_sent.lower())
    example = ins(questions.numericalize(questions.pad([sent]), device=args.gpu, train=False))
    model.eval()
    scores = model(example)
    # get the predicted relations
    top_scores, top_indices = torch.max(scores, dim=1)  # shape: (batch_size, 1)
    top_index = top_indices.cpu().data.numpy()[0]
    predicted_relation = index2rel[top_index]
    return predicted_relation

print("question: {}".format(input_sent))
print("predicted relation: {}".format(get_relation(input_sent)))
