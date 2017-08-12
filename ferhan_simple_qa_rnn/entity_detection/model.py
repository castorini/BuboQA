from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch

class EntityDetection(nn.Module):

    def __init__(self, config):
        super(EntityDetection, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        if self.config.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=config.d_embed, hidden_size=config.d_hidden,
                              num_layers=config.n_layers, dropout=config.dropout_prob,
                              bidirectional=config.birnn)
        else:
            self.rnn = nn.LSTM(input_size=config.d_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.relu = nn.ReLU()
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.hidden2tag = nn.Sequential(
                        nn.Linear(seq_in_size, seq_in_size),
                        nn.BatchNorm1d(seq_in_size),
                        self.relu,
                        self.dropout,
                        nn.Linear(seq_in_size, config.n_out)
        )

    def forward(self, batch):
        # shape of batch (sequence length, batch size)
        inputs = self.embed(batch.question) # shape (sequence length, batch_size, dimension of embedding)
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        if self.config.rnn_type.lower() == 'gru':
            h0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, ht = self.rnn(inputs, h0)
        else:
            h0 = c0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        # shape of `outputs` - (sequence length, batch size, hidden size X num directions)
        tags = self.hidden2tag(outputs.view(-1, outputs.size(2)))
        # print(tags)
        scores = F.log_softmax(tags)
        return scores

