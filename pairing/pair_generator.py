import random
from scipy.sparse import vstack

def generate_pairs(X, labels, num_pairs):
    label_to_indices = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    pairs_X = []
    pairs_y = []

    for _ in range(num_pairs):
        if random.random() < 0.5:
            lbl = random.choice(list(label_to_indices.keys()))
            i, j = random.sample(label_to_indices[lbl], 2)
            pairs_X.append(abs(X[i] - X[j]))
            pairs_y.append(1)
        else:
            lbl1, lbl2 = random.sample(list(label_to_indices.keys()), 2)
            i = random.choice(label_to_indices[lbl1])
            j = random.choice(label_to_indices[lbl2])
            pairs_X.append(abs(X[i] - X[j]))
            pairs_y.append(0)

    return vstack(pairs_X), pairs_y
