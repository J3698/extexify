import torch
import subproccess
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
    subproccess.run(["wget", "--no-check-certificate", \
                     f"https://docs.google.com/uc?export=download&id={dX}", \
                     "-O", "dataX.npy"]
dY = "1sArcVn6WCftYdtRmziy8t7V6D8Dr3Vhd"
if not os.path.exists("./dataY.npy"):
    subproccess.run(["wget", "--no-check-certificate", \
                     f"https://docs.google.com/uc?export=download&id={dY}", \
                     "-O", "dataY.npy"]

dataset = ExtexifyDataset("./dataX.npy", "./dataY.npy")
dataloader = DataLoader(dataset, batch_size = 16,\
                        shuffle = True, collate_fn = collate)
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
total = 0
correct = 0
i = 0
for x, y in tqdm(dataloader):
    i += 1
    optimizer.zero_grad()

    out = model(x)
    loss = criterion(out, y)

    loss.backward()
    optimizer.step()

    total += len(x)
    correct += (torch.argmin(out) == y).sum()
    if i % 5 == 0:
        print(f"{correct / total:.2f}")
        print(correct)


