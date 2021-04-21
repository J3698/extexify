import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from prep_data import ExtexifyDataset, download_dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from models import *


def collate(batch):
    x, y = zip(*batch)
    packed_x = pack_sequence(x, enforce_sorted = False)
    y = torch.tensor(y)
    return packed_x, y


download_dataset()
batch_size = 512 if torch.cuda.is_available() else 2
dataset = ExtexifyDataset("./dataX.npy", "./dataY.npy")
dataloader = DataLoader(dataset, batch_size = batch_size,\
                        shuffle = True, collate_fn = collate)

model = Model()
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
for epoch in range(10):
    total = 0
    correct = 0
    correct5 = 0
    i = 0
    bar = tqdm(dataloader)
    for x, y in bar:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        i += 1
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total += len(x)

        c = (torch.argmax(out, dim = 1) == y).sum().item()

        top5 = torch.topk(out, 5, dim = 1).indices
        assert top5.shape == (x.shape[0], 5)
        c5 = torch.any(top5 == y[:, None], dim = 1).sum().item()

        correct += c
        correct5 += c5

        bar.set_postfix({"correct": c, "top1": correct / total,
                         "top5": correct5 / total})
