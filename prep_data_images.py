import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from PIL import Image, ImageDraw
from tqdm import tqdm


def main():
    dataset = ExtexifyDataset()
    for i in tqdm(range(len(dataset))):
        stroke, y = dataset[i]
        if not os.path.exists(f"images/{y.item()}"):
            os.mkdir(f"images/{y.item()}")

        image = Image.new("L", (64, 64), color = 0)
        draw = ImageDraw.Draw(image)
        for j in range(len(stroke) - 1):
            if j != 0 and stroke[j, 2] != 1:
                if stroke[j, 2] != 0:
                    print(stroke[j, 2])
                p1 = (stroke[j, :2].numpy() * 64).tolist()
                p2 = (stroke[j + 1, :2].numpy() * 64).tolist()
                draw.line(p1 + p2, fill=255, width=1)
        image.save(f"images/{y.item()}/{i}.png")

    print("Created dataset")


def collate(batch):
    x, y = zip(*batch)
    packed_x = pack_sequence(x, enforce_sorted = False)
    y = torch.tensor(y)
    return packed_x, y


def dataloaders(batch_size):
    download_dataset()
    dataset = ExtexifyDataset("./dataX.npy", "./dataY.npy")
    dataset_train, dataset_val, dataset_test = \
        dataset.split_train_val_test(0.7, 0.2, 0.1)

    dataloader_train = DataLoader(dataset_train, batch_size = batch_size,\
                                  shuffle = True, collate_fn = collate,
                                  num_workers = os.cpu_count())
    dataloader_val = DataLoader(dataset_val, batch_size = batch_size,\
                                shuffle = False, collate_fn = collate)
    dataloader_test = DataLoader(dataset_test, batch_size = batch_size,\
                                 shuffle = False, collate_fn = collate)

    return dataloader_train, dataloader_val, dataloader_test


class ExtexifyDataset(torch.utils.data.Dataset):
    def __init__(self, strokes_file = "data_processed/dataX.npy",\
                       labels_file = "data_processed/dataY.npy",\
                       strokes_labels = None):
        if strokes_labels is not None:
            self.strokes, self.labels = strokes_labels
        else:
            strokes = np.load(strokes_file, allow_pickle = True)
            labels = np.load(labels_file, allow_pickle = True)
            print("Loaded file")
            assert all([i.shape[1] == 4 for s in strokes for i in s])

            self.strokes = [torch.tensor(np.delete(np.vstack(inst), 2, axis = 1)).float() \
                            for inst in tqdm(strokes)]
            for i, stroke in tqdm(enumerate(self.strokes), total = len(self.strokes)):
                stroke[:, :2] -= stroke[:, :2].min(dim = 0).values
                stroke[:, :2] /= (stroke[:, :2].max(dim = 0).values + 1e-15)

            print("Processed strokes")

            labels_dict = {l: i for i, l in enumerate(sorted(set(labels)))}
            self.labels = torch.tensor([labels_dict[i] for i in labels])
            print("Processed labels")

            assert len(self.strokes) == len(self.labels)


    def __len__(self):
        return len(self.strokes)


    def __getitem__(self, idx):
        return self.strokes[idx], self.labels[idx]


    def split_train_val_test(self, percent_train, percent_val, percent_test):
        total = percent_train + percent_val + percent_test
        assert abs(total - 1) < 1e-9, total
        assert 0 < percent_train and 0 < percent_val and 0 < percent_test

        train, rest = self.split(percent_train)
        val, test = rest.split(percent_val / (percent_test + percent_val))
        return train, val, test

    def split(self, percent_train):
        x_train, x_test, y_train, y_test = \
            train_test_split(self.strokes, self.labels, \
                             train_size = percent_train, random_state = 13)
        train = ExtexifyDataset(strokes_labels = [x_train, y_train])
        test = ExtexifyDataset(strokes_labels = [x_test, y_test])
        return train, test


def download(filename, drivelink):
    if not os.path.exists(filename):
        confirm_cmd = ["wget", "--quiet", "--save-cookies", "/tmp/cookies.txt", \
                       "--keep-session-cookies", "--no-check-certificate", \
                       f'https://docs.google.com/uc?export=download&id={drivelink}', '-O-']
        confirm = subprocess.Popen(confirm_cmd, stdout = subprocess.PIPE)
        sed_cmd = ["sed", "-r", "-n", 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p']
        sed = subprocess.Popen(sed_cmd, stdin = confirm.stdout, stdout = subprocess.PIPE)
        confirm.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        output = sed.communicate()[0].decode('utf-8').replace("\n", "")
        link = f"https://docs.google.com/uc?export=download&confirm={output}&id={drivelink}"
        cmd = ["wget", "--load-cookies", "/tmp/cookies.txt", link, "-O", filename]
        subprocess.run(cmd)
        subprocess.run(["rm", "-rf", "/tmp/cookies.txt"])


def download_dataset():
    download("./dataX.npy", "18KMxHJujq8Nb3SIMMFPHvTvQ3oBJtqXZ")
    download("./dataY.npy", "1sArcVn6WCftYdtRmziy8t7V6D8Dr3Vhd")




if __name__ == "__main__":
    main()


