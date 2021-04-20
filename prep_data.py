import numpy as np
import torch
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    main()


