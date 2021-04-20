import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class Model(nn.Module):
    def __init__(self, num_classes = 1098, hidden_size = 64):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size, 2, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(2 * hidden_size, num_classes)

        self.num_classes = num_classes
        self.hidden_size = hidden_size


    def forward(self, x):
        h, _ = self.lstm(x)
        h_unp, lens_unp = pad_packed_sequence(h, batch_first = True)
        B, T, _ = h_unp.shape
        lasts = h_unp[torch.arange(B), lens_unp - 1]

        assert lasts.shape == (B, self.hidden_size * 2)
        return self.linear(lasts)
