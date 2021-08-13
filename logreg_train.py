import sys
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from utils import StandardScaler, train_test_split

np.random.seed(5)


class LogisticRegression(object):
    def __init__(self, teta=0.1, max_iter=50, l1l2_coef=0, weights=None, num_classes=None):
        self.eta = teta
        self.max_iter = max_iter
        self.l1l2_coef = l1l2_coef
        self.weights = weights
        self.num_classes = num_classes
        self.errors = []
        self.costs = []

    def fit(self, X, y, sample_weight=None):
        self.num_classes = np.unique(y).tolist()
        X_factor = np.insert(X, 0, 1, axis=1)
        m = X_factor.shape[0]

        self.weights = sample_weight
        if not self.weights:
            self.weights = np.zeros(X_factor.shape[1] * len(self.num_classes))
        self.weights = self.weights.reshape(len(self.num_classes), X_factor.shape[1])

        y_factor = np.zeros((len(y), len(self.num_classes)))
        for i in range(0, len(y)):
            y_factor[i, self.num_classes.index(y[i])] = 1
        for i in tqdm(range(0, self.max_iter)):
            predictions = self.net_input(X_factor).T

            left = y_factor.T.dot(np.log(predictions))
            right = (1 - y_factor).T.dot(np.log(1 - predictions))

            r1 = (self.l1l2_coef / (2 * m)) * sum(sum(self.weights[:, 1:] ** 2))
            cost = (-1 / m) * sum(left + right) + r1
            self.costs.append(cost)
            self.errors.append(sum(y != self.predict(X)))

            r2 = (self.l1l2_coef / m) * self.weights[:, 1:]
            self.weights = self.weights - (self.eta * (1 / m) * (predictions - y_factor).T.dot(X_factor) + np.insert(r2, 0, 0, axis=1))
        return self

    def net_input(self, X):
        return self.sigmoid(self.weights.dot(X.T))

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        predictions = self.net_input(X).T
        return [self.num_classes[x] for x in predictions.argmax(1)]

    def save_weights(self, scaler, filename='weights.csv'):
        with open(filename, 'w+') as f:
            for i in range(0, len(self.num_classes)):
                f.write(f'{self.num_classes[i]},')
            f.write('mean,std\n')

            for j in range(0, self.weights.shape[1]):
                for i in range(0, self.weights.shape[0]):
                    f.write(f'{self.weights[i][j]},')
                f.write(f'{scaler.mean[j - 1] if j > 0 else ""},{scaler.std[j - 1] if j > 0 else ""}\n')
        return self

    def sigmoid(self, z):
        result = 1.0 / (1.0 + np.exp(-z))
        return result


if __name__ == '__main__':
    dataset = Dataset(sys.argv[1])
    df = dataset.df
    df = df.dropna(subset=['defense_against_the_dark_arts', 'charms', 'herbology', 'divination', 'muggle_studies'])
    X = np.array(df.values[:, [9, 17, 8, 10, 11]], dtype=float)
    y = df.values[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    regressor = LogisticRegression(teta=0.01, max_iter=50, l1l2_coef=10)
    regressor.fit(X_train_std, y_train)

    y_pred = regressor.predict(X_test_std)
    print(f'Accuracy by sklearn: {accuracy_score(y_test, y_pred)}')

    regressor.save_weights(scaler)
