"""
This is a very simple feed forward NN with 2 i/p features, 1 hidden layer with 2 neurons, and 1 output layer.
Therefore, the param sizes: W1 = 2x2, b1 =2, W2 = 1x2, b = 2
Everything written from scratch
code motivation: One-Fourth Labs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class SimpleFFNN:

    def __init__(self):
        # 1st layer weight init
        self.w111 = np.random.randn()  # weight associated with layer 1, neuron 1 and inp x1
        self.w112 = np.random.randn()  # weight associated with layer 1, neuron 1 and inp x2
        self.w121 = np.random.randn()  # weight associated with layer 1, neuron 2 and inp x1
        self.w122 = np.random.randn()  # weight associated with layer 1, neuron 2 and inp x2
        self.b1 = 0  # bias associated with neuron 1
        self.b2 = 0  # bias associated with neuron 2
        # 2nd layer weight init
        self.w211 = np.random.randn()  # weight associated with layer 2, neuron 1 and inp x1
        self.w212 = np.random.randn()  # weight associated with layer 2, neuron 1 and inp x2
        self.b3 = 0  # bias associated with neuron 1 of op

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward_pass(self, x):
        self.x1, self.x2 = x
        # computations of 1st layer
        self.a11 = self.w111 * self.x1 + self.w112 * self.x2 + self.b1
        self.a12 = self.w121 * self.x1 + self.w122 * self.x2 + self.b2
        self.h11 = self.sigmoid(self.a11)
        self.h12 = self.sigmoid((self.a12))

        # computation of last layer
        self.a2 = self.w211 * self.h11 + self.w212 * self.h12 + self.b3
        self.h3 = self.sigmoid(self.a2)
        return self.h3

    def grad(self, x, y):
        y_hat = self.forward_pass(x)

        self.dw212 = (y_hat - y) * y_hat * (1 - y_hat) * self.h12
        self.dw211 = (y_hat - y) * y_hat * (1 - y_hat) * self.h11
        self.db3 = (y_hat - y) * y_hat * (1 - y_hat)

        self.dw111 = (y_hat - y) * y_hat * (1 - y_hat) * self.w211 * self.h11 * (1 - self.h11) * self.x1
        self.dw112 = (y_hat - y) * y_hat * (1 - y_hat) * self.w211 * self.h11 * (1 - self.h11) * self.x2
        self.db1 = (y_hat - y) * y_hat * (1 - y_hat) * self.w211 * self.h11 * (1 - self.h11)

        self.dw121 = (y_hat - y) * y_hat * (1 - y_hat) * self.w212 * self.h12 * (1 - self.h12) * self.x1
        self.dw122 = (y_hat - y) * y_hat * (1 - y_hat) * self.w212 * self.h12 * (1 - self.h12) * self.x2
        self.db2 = (y_hat - y) * y_hat * (1 - y_hat) * self.w212 * self.h12 * (1 - self.h12)

    def fit(self, X, Y, epochs=1, learning_rate=1, init=True, display_loss=False):
        if init:
            # 1st layer weight init
            self.w111 = np.random.randn()  # weight associated with layer 1, neuron 1 and inp x1
            self.w112 = np.random.randn()  # weight associated with layer 1, neuron 1 and inp x2
            self.w121 = np.random.randn()  # weight associated with layer 1, neuron 2 and inp x1
            self.w122 = np.random.randn()  # weight associated with layer 1, neuron 2 and inp x2
            self.b1 = 0  # bias associated with neuron 1
            self.b2 = 0  # bias associated with neuron 2
            # 2nd layer weight init
            self.w211 = np.random.randn()  # weight associated with layer 2, neuron 1 and inp x1
            self.w212 = np.random.randn()  # weight associated with layer 2, neuron 1 and inp x2
            self.b3 = 0  # bias associated with neuron 1 of op
        if display_loss:
            loss = {}
        for epoch in tqdm(range(epochs)):
            dw111, dw112, dw121, dw122, db1, db2, dw211, dw212, db3 = [0] * 9
            for x, y in zip(X, Y):
                self.grad(x, y)
                dw111 += self.dw111
                dw112 += self.dw112
                dw121 += self.dw121
                dw122 += self.dw122
                db1 += self.db1
                db2 += self.db2
                dw211 += self.dw211
                dw212 += self.dw212
                db3 += self.db3

            m = X.shape[1]
            self.w111 = self.w111 - learning_rate * dw111 / m
            self.w112 = self.w112 - learning_rate * dw112 / m
            self.w121 = self.w121 - learning_rate * dw121 / m
            self.w122 = self.w122 - learning_rate * dw122 / m
            self.b1 = self.b1 - learning_rate * db1 / m
            self.b2 = self.b2 - learning_rate * db2 / m
            self.w211 = self.w211 - learning_rate * dw211 / m
            self.w212 = self.w212 - learning_rate * dw212 / m
            self.b3 = self.b3 - learning_rate * db3 / m

            if display_loss:
                Y_pred = self.predict(X)
                loss[epoch] = mean_squared_error(Y_pred, Y)
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

    def predict(self, X):
        Y_hat = []
        for x in X:
            y_hat = self.forward_pass(x)
            Y_hat.append(y_hat)
        return np.array(Y_hat)


def preapare_random_data():
    data, labels, = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
    labels_orig = labels
    labels = np.mod(labels_orig, 2)
    X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
    return X_train, X_val, Y_train, Y_val


if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = preapare_random_data()
    ffn = SimpleFFNN()
    ffn.fit(X_train, Y_train, epochs=2000, learning_rate=0.01, display_loss=True)
    Y_pred_train = ffn.predict(X_train)
    Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()
    Y_pred_val = ffn.predict(X_val)
    Y_pred_binarised_val = (Y_pred_val >= 0.5).astype("int").ravel()
    accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_binarised_val, Y_val)

    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))