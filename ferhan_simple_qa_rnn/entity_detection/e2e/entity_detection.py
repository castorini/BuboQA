import torch
from torchtext import data
from simple_qa_ner import SimpleQADataset
import os
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
gpu = -1

if torch.cuda.is_available():
    gpu = 0
    torch.cuda.set_device(gpu)

questions = data.Field(lower=True, sequential=True)
labels = data.Field(sequential=True)
train, dev, test = SimpleQADataset.splits(questions, labels)
questions.build_vocab(train, dev, test)
labels.build_vocab(train, dev, test)

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


trained_model = "model/model.pt"
model = torch.load(trained_model, map_location=lambda storage,location: storage.cuda(gpu))

index2tag = np.array(labels.vocab.itos)
index2word = np.array(questions.vocab.itos)

input_sent = "where was sasha vujacic born?"


class ins(object):
    def __init__(self, question):
      self.question = question

def get_span(label):
    span = []
    st = 0
    en = 0
    flag = False
    for k, l in enumerate(label):
        if l == 'I' and flag == False:
            st = k
            flag = True
        if l != 'I' and flag == True:
            flag = False
            en = k
            span.append((st, en))
            st = 0
            en = 0
    if st != 0 and en == 0:
        en = k
        span.append((st, en))
    return span



def get_query_text(input_sent):
    sent = tokenizer.tokenize(input_sent.lower())
    example = ins(questions.numericalize(questions.pad([sent]), device=gpu, train=False))
    model.eval()
    scores = model(example)
    index_tag = np.transpose(torch.max(scores, 1)[1].cpu().data.numpy())
    tag_array = index2tag[index_tag][0]
    spans = get_span(tag_array)
    query_tokens = []
    for span in spans:
        query_tokens.append(" ".join(sent[span[0]:span[1]]))
    return query_tokens

print(get_query_text(input_sent))
