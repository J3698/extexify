import torch
import torch.optim as optim
import torch.nn as nn
from prep_data import dataloaders
from tqdm import tqdm
from models import *
import torch.optim.lr_scheduler as lr_scheduler


batch_size = 512 if torch.cuda.is_available() else 2
epochs = 20
step_size = 21


def main():
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = dataloaders(batch_size)

    models = [Model1()]
    for model in models:
        print(model.num_layers, model.hidden_size)
        optimizer = optim.Adam(model.parameters())
        scheduler = lr_scheduler.StepLR(optimizer, step_size)
        run_name = f"{model.hidden_size}-{model.num_layers}"
        train_model(run_name, model, criterion, optimizer, \
                    scheduler, epochs, train_loader, val_loader, test_loader)
        print()


def train_model(run_name, model, criterion, optimizer, scheduler,\
                epochs, train_loader, val_loader, test_loader):
    if torch.cuda.is_available():
        model.cuda()

    best_top5 = 0
    for e in range(epochs):
        _, top5 = validate(model, val_loader)
        if top5 > best_top5:
            best_top5 = top5
            save(f"{run_name}.pt", model, optimizer, scheduler, e)

        train_epoch(model, optimizer, criterion, train_loader, scheduler)


def save(filename, model, optimizer, scheduler, epoch):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }, filename)


def train_epoch(model, optimizer, criterion, train_loader, scheduler):
    total = 0
    correct = 0
    correct5 = 0

    model.train()

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
        correct += topk_correct(out, y, 1)
        correct5 += topk_correct(out, y, 5)
        update_bar(bar, correct, correct5, total)

    scheduler.step()


@torch.no_grad()
def validate(model, eval_loader):
    total = 0
    correct = 0
    correct5 = 0

    model.eval()

    bar = tqdm(eval_loader)
    for x, y in bar:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        out = model(x)

        total += len(y)
        correct += topk_correct(out, y, 1)
        correct5 += topk_correct(out, y, 5)
        update_bar(bar, correct, correct5, total)

    return correct / total, correct5 / total


def update_bar(bar, correct, correct5, total):
    postfix = {"top1": 100 * correct / total,\
               "top5": 100 * correct5 / total}
    bar.set_postfix(postfix)


def topk_correct(out, y, k):
    topk = torch.topk(out, k, dim = 1).indices
    return torch.any(topk == y[:, None], dim = 1).sum().item()


if __name__ == "__main__":
    main()


