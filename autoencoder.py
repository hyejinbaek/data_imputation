import numpy as np
import random
import itertools
import pandas as pd

np.random.seed(37)
random.seed(37)

size = 1_000

X_0 = np.random.normal(0, 1, size=size)
X_1 = 1.1 + 4 * X_0 + np.random.normal(0, 1, size=size)
X_2 = 2.3 - 0.5 * X_0 + np.random.normal(0, 1, size=size)

X = np.hstack([X_0.reshape(-1, 1), X_1.reshape(-1, 1), X_2.reshape(-1, 1)])

#print(X.shape)


def make_missing(X, frac=0.1):
    n = int(frac * X.shape[0] * X.shape[1])

    rows = list(range(X.shape[0]))
    cols = list(range(X.shape[1]))

    coordinates = list(itertools.product(*[rows, cols]))
    random.shuffle(coordinates)
    coordinates = coordinates[:n]

    M = np.copy(X)

    for r, c in coordinates:
        M[r, c] = np.nan

    return pd.DataFrame(M, columns=[f'X_{i}' for i in range(X.shape[1])]), coordinates

df, coordinates = make_missing(X)
