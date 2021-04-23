import torch
import torch.optim as optim
import torch.nn as nn
from prep_data import dataloaders
from tqdm import tqdm
from models import *


batch_size = 512 if torch.cuda.is_available() else 2
epochs = 10

def main():
    train_loader, val_loader, test_loader = dataloaders(batch_size)
    model = Model()
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for e in range(epochs):
        validate(model, val_loader)
        train(model, optimizer, criterion, train_loader)

def train(model, optimizer, criterion, train_loader):
    total = 0
    correct = 0
    correct5 = 0
    i = 0
    bar = tqdm(train_loader)
    for x, y in bar:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total += len(y)
        c = (torch.argmax(out, dim = 1) == y).sum().item()
        top5 = torch.topk(out, 5, dim = 1).indices
        c5 = torch.any(top5 == y[:, None], dim = 1).sum().item()
        correct += c
        correct5 += c5

        postfix = {"correct": c,\
                   "top1": correct / total,\
                   "top5": correct5 / total}
        bar.set_postfix(postfix)

@torch.no_grad()
def validate(model, eval_loader):
    total = 0
    correct = 0
    correct5 = 0
    i = 0
    bar = tqdm(eval_loader)

    for x, y in bar:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        out = model(x)

        total += len(y)
        c = (torch.argmax(out, dim = 1) == y).sum().item()
        top5 = torch.topk(out, 5, dim = 1).indices
        c5 = torch.any(top5 == y[:, None], dim = 1).sum().item()
        correct += c
        correct5 += c5

        postfix = {"correct": c,\
                   "top1": correct / total,\
                   "top5": correct5 / total}
        bar.set_postfix(postfix)


if __name__ == "__main__":
    main()


