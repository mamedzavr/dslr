import numpy as np
from describe import calc_mean, calc_std

np.random.seed(5)


class StandardScaler():
    def __init__(self, mean=np.array([]), std=np.array([])):
        self.mean = mean
        self.std = std

    def fit(self, X):
        for i in range(0, X.shape[1]):
            self.mean = np.append(self.mean, calc_mean(X[:, i]))
            self.std = np.append(self.std, calc_std(X[:, i]))

    def transform(self, X):
        return (X - self.mean) / self.std


def train_test_split(X, y, test_size=0.3):
    p = np.random.permutation(len(X))

    X_offset = int(len(X) * test_size)
    y_offset = int(len(y) * test_size)

    X_train = X[p][X_offset:]
    X_test = X[p][:X_offset]

    y_train = y[p][y_offset:]
    y_test = y[p][:y_offset]
    return X_train, X_test, y_train, y_test
