import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess
import os

def main():
    dataset = ExtexifyDataset("./dataX.npy", "./dataY.npy")
    X, Y = dataset[1]
    a = np.ones((100, 100))
    for x, y, t, z in X:
        a[int(y * 99), int(x * 99)] = 0
    plt.imshow(a)
    plt.show()


class ExtexifyDataset(torch.utils.data.Dataset):
    def __init__(self, strokes_file, labels_file):
        strokes = np.load(strokes_file, allow_pickle = True)
        labels = np.load(labels_file, allow_pickle = True)
        print("Loaded data")

        assert all([i.shape[1] == 4 for s in strokes for i in s])
        self.strokes = [torch.tensor(np.vstack(inst)).float() for inst in strokes]
        for stroke in self.strokes:
            stroke[:, :3] -= stroke[:, :3].min(dim = 0).values
            stroke[:, :3] /= (stroke[:, :3].max(dim = 0).values + 1e-15)
        print("Processed strokes")

        labels_dict = {l: i for i, l in enumerate(sorted(set(labels)))}
        self.labels = torch.tensor([labels_dict[i] for i in labels])
        print("Processed labels")

        assert len(self.strokes) == len(self.labels)

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        return self.strokes[idx], self.labels[idx]


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


