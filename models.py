import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes = 1098, hidden_size = 32):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size, 1, batch_first = True)
        self.linear = nn.Linear(hidden_size, num_classes)

        self.num_classes = num_classes
        self.hidden_size = hidden_size


    def forward(self, x):
        h, _ = self.lstm(x)
        h_unp, lens_unp = pad_packed_sequence(h, batch_first = True)
        B, T, _ = h_unp.shape
        lasts = h_unp[torch.arange(B), lens_unp - 1]

        assert lasts.shape == (B, self.hidden_size)
        return self.linear(lasts)
