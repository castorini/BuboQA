import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class RelationPrediction(nn.Module):
    def __init__(self, config):
        super(RelationPrediction, self).__init__()
        self.config = config
        target_size = config.rel_label
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        if config.train_embed == False:
            self.embed.weight.requires_grad = False
        if config.relation_prediction_mode.upper() == "GRU":
            self.gru = nn.GRU(input_size=config.input_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_layer,
                               dropout=config.rnn_dropout,
                               bidirectional=True)
            self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
            self.relu = nn.ReLU()
            self.hidden2tag = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.BatchNorm1d(config.hidden_size * 2),
                self.relu,
                self.dropout,
                nn.Linear(config.hidden_size * 2, target_size)
            )
        if config.relation_prediction_mode.upper() == "LSTM":
            self.lstm = nn.LSTM(input_size=config.input_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_layer,
                               dropout=config.rnn_dropout,
                               bidirectional=True)
            self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
            self.relu = nn.ReLU()
            self.hidden2tag = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.BatchNorm1d(config.hidden_size * 2),
                self.relu,
                self.dropout,
                nn.Linear(config.hidden_size * 2, target_size)
            )
        if config.relation_prediction_mode.upper() == "CNN":
            input_channel = 1
            Ks = 3
            self.conv1 = nn.Conv2d(input_channel, config.output_channel, (2, config.words_dim), padding=(1, 0))
            self.conv2 = nn.Conv2d(input_channel, config.output_channel, (3, config.words_dim), padding=(2, 0))
            self.conv3 = nn.Conv2d(input_channel, config.output_channel, (4, config.words_dim), padding=(3, 0))
            self.dropout = nn.Dropout(config.cnn_dropout)
            self.fc1 = nn.Linear(Ks * config.output_channel, target_size)


    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        batch_size = text.size()[1]
        x = self.embed(text)
        if self.config.relation_prediction_mode.upper() == "LSTM":
            # h0 / c0 = (layer*direction, batch_size, hidden_dim)
            if self.config.cuda:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size).cuda())
                c0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size))
                c0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size))
            # output = (sentence length, batch_size, hidden_size * num_direction)
            # ht = (layer*direction, batch, hidden_dim)
            # ct = (layer*direction, batch, hidden_dim)
            outputs, (ht, ct) = self.lstm(x, (h0, c0))
            tags = self.hidden2tag(ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1))
            scores = F.log_softmax(tags)
            return scores
        elif self.config.relation_prediction_mode.upper() == "GRU":
            if self.config.cuda:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size))
            outputs, ht = self.gru(x, h0)

            tags = self.hidden2tag(ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1))
            scores = F.log_softmax(tags)
            return scores
        elif self.config.relation_prediction_mode.upper() == "CNN":
            x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
            x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
            # (batch, channel_output, ~=sent_len) * Ks
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
            # (batch, channel_output) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)
            x = self.dropout(x)
            logit = self.fc1(x)  # (batch, target_size)
            scores = F.log_softmax(logit)
            return scores
        else:
            print("Unknown Mode")
            exit(1)



