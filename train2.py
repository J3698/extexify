import os
import torch
import torch.optim as optim
import torch.nn as nn
from prep_data import dataloaders
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


batch_size = 512 if torch.cuda.is_available() else 2
epochs = 20
step_size = 21


def main():
    # loss function
    criterion = nn.CrossEntropyLoss()

    # datasets
    dataset_train = ImageFolder("./images_data32/train", transform = ToTensor())
    dataset_val = ImageFolder("./images_data32/val", transform = ToTensor())
    dataset_test = ImageFolder("./images_data32/test", transform = ToTensor())

    # for loading data into batches
    train_loader = DataLoader(dataset_train, batch_size = batch_size,\
                              shuffle = True, num_workers = os.cpu_count())
    val_loader = DataLoader(dataset_val, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)

    run_name = "Test"
    model = Model()
    optimizer = optim.Adam(model.parameters(), weight_decay = 1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size)

    train_model(run_name, model, criterion, optimizer, \
                scheduler, epochs, train_loader, val_loader, test_loader)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(256),
                nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(512),

                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(512, 1098)
        )

    def forward(self, x):
        return self.layers(x)


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
            print("Saved model")

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


