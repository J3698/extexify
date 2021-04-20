import torch
import subprocess
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from prep_data import ExtexifyDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import os
from models import *



def collate(batch):
    x, y = zip(*batch)
    packed_x = pack_sequence(x, enforce_sorted = False)
    y = torch.tensor(y)
    return packed_x, y


dX = "18KMxHJujq8Nb3SIMMFPHvTvQ3oBJtqXZ"
if not os.path.exists("./dataX.npy"):
    subprocess.run(["wget", "--no-check-certificate", \
                     f"https://docs.google.com/uc?export=download&id={dX}", \
                     "-O", "dataX.npy"])
dY = "1sArcVn6WCftYdtRmziy8t7V6D8Dr3Vhd"
if not os.path.exists("./dataY.npy"):
    subprocess.run(["wget", "--no-check-certificate", \
                     f"https://docs.google.com/uc?export=download&id={dY}", \
                     "-O", "dataY.npy"])

dataset = ExtexifyDataset("./dataX.npy", "./dataY.npy")
dataloader = DataLoader(dataset, batch_size = 512,\
                        shuffle = True, collate_fn = collate)
model = Model()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
for epoch in range(10):
    total = 0
    correct = 0
    i = 0
    bar = tqdm(dataloader)
    for x, y in bar:
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
        correct += c

        bar.set_postfix({"correct": c, "acc": correct / total})





