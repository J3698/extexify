import numpy as np
from sklearn.model_selection import train_test_split


def load_data(val_ratio, test_ratio, strokes_file, labels_file):
    strokes = np.load(strokes_file, allow_pickle = True)
    labels = np.load(labels_file, allow_pickle = True)
    print("Loaded data")

    assert all([i.shape[1] == 4 for s in strokes for i in s])
    X = [np.delete(np.vstack(inst), 2, axis = 1).astype(float) for inst in strokes]
    for i, stroke in enumerate(X):
        stroke[:, :2] -= stroke[:, :2].min(axis = 0)
        stroke[:, :2] /= (stroke[:, :2].max(axis = 0) + 1e-15)
    print("Processed strokes")

    labels_dict = {l: i for i, l in enumerate(sorted(set(labels)))}
    Y = np.array([labels_dict[i] for i in labels])
    print("Processed labels")

    x_train, x_eval, y_train, y_eval = \
        train_test_split(X, Y, \
                         train_size = 1 - val_ratio - test_ratio, \
                         random_state = 13)
    x_val, x_test, y_val, y_test = \
        train_test_split(x_eval, y_eval, \
                         train_size = val_ratio / (val_ratio + test_ratio), \
                         random_state = 13)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert len(x_test) == len(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test
