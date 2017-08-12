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
        if self.config.birnn:
            self.hidden2tag = nn.Linear(config.d_hidden * 2, config.n_out)
        else:
            self.hidden2tag = nn.Linear(config.d_hidden, config.n_out)



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
        tags = F.softmax(self.hidden2tag(outputs.view(-1, outputs.size(2))))
        #print(tags)
        return tags

