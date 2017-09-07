import torch
from torchtext import data
from simple_qa_relation import SimpleQaRelationDataset
import os
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
gpu = -1

if torch.cuda.is_available():
    gpu = 0
    torch.cuda.set_device(gpu)

questions = data.Field(lower=True, sequential=True)
relations = data.Field(sequential=False)
train, dev, test = SimpleQaRelationDataset.splits(questions, relations)
questions.build_vocab(train, dev, test)
relations.build_vocab(train, dev, test)

vector_cache = 'vector_cache/input_vectors.pt'
data_cache = 'data_cache'
word_vectors = 'glove.42B'
d_embed = 300

if os.path.isfile(vector_cache):
    questions.vocab.vectors = torch.load(vector_cache)
else:
    questions.vocab.load_vectors(wv_dir=data_cache, wv_type=word_vectors, wv_dim=d_embed)
    os.makedirs(os.path.dirname(vector_cache), exist_ok=True)
    torch.save(questions.vocab.vectors, vector_cache)


trained_model = "rel_model.pt"
model = torch.load(trained_model, map_location=lambda storage,location: storage.cuda(gpu))

index2rel = np.array(relations.vocab.itos)

input_sent = "where was sasha vujacic born?"

class ins(object):
    def __init__(self, question):
      self.question = question


def get_relation(input_sent):
    sent = tokenizer.tokenize(input_sent.lower())
    example = ins(questions.numericalize(questions.pad([sent]), device=gpu, train=False))
    model.eval()
    scores = model(example)
    # get the predicted relations
    top_scores, top_indices = torch.max(scores, dim=1)  # shape: (batch_size, 1)
    top_index = top_indices[0][0]
    predicted_relation = index2rel[top_index]
    return predicted_relation

print(get_relation(input_sent))
