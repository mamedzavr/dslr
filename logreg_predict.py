import sys
import pandas as pd
import numpy as np
from logreg_train import StandardScaler, LogisticRegression

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    df = df.fillna(method='ffill')
    X = np.array(df.values[:, [9, 17, 8, 10, 11]], dtype=float)

    df = pd.read_csv(sys.argv[2])
    classes = list(df)[:4]
    mean = df.values[1:, 4]
    std = df.values[1:, 5]
    weights = df.values[:, :4].T

    scaler = StandardScaler(mean, std)
    X_std = scaler.transform(X)

    lr = LogisticRegression(weights=weights, num_classes=classes)

    y_pred = lr.predict(X_std)

    with open("houses.csv", 'w+') as f:
        f.write('Index,Hogwarts House\n')
        for i in range(0, len(y_pred)):
            f.write(f'{i},{y_pred[i]}\n')

