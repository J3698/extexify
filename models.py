import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, hidden_size = 64, num_layers = 2, bidirectional = True, num_classes = 1098):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden_size, num_layers, bidirectional = bidirectional, \
                        batch_first = True)
        self.linear = nn.Linear((2 if bidirectional else 1) * hidden_size, num_classes)

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers


    def forward(self, x):
        h, _ = self.lstm(x)
        h_unp, lens_unp = pad_packed_sequence(h, batch_first = True)
        B, T, _ = h_unp.shape
        lasts = h_unp[torch.arange(B), lens_unp - 1]

        assert lasts.shape == (B, self.hidden_size * 2)
        return self.linear(lasts)

class Model1(nn.Module):
    def __init__(self, hidden_size = 128, num_layers = 2, bidirectional = True, \
                 num_classes = 1098, dropout = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size, num_layers, bidirectional = bidirectional, \
                            batch_first = True, dropout = dropout)
        self.linear = nn.Linear((2 if bidirectional else 1) * hidden_size, num_classes)

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


    def forward(self, x):
        h, _ = self.lstm(x)
        h_unp, lens_unp = pad_packed_sequence(h, batch_first = True)
        B, T, _ = h_unp.shape
        lasts = h_unp[torch.arange(B), lens_unp - 1]

        assert lasts.shape == (B, self.hidden_size * 2)
        return self.linear(lasts)
