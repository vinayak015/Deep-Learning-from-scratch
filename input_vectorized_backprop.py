import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs


class FullVectorizedNN:
    """
    This class vectorizes the weights and biases as well as the input
    """

    def __init__(self):
        self.W1 = np.random.randn(2, 2)
        self.W2 = np.random.randn(2, 4)
        self.B1 = np.zeros((1, 2))
        self.B2 = np.zeros((1, 4))

    def sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))  # Nx2

    def softmax(self, A):
        exps = np.exp(A)  # Nx4
        return exps / np.sum(exps, axis=1).reshape(-1, 1)  # Nx1

    def forward_pass(self, X):
        self.A1 = np.matmul(X, self.W1) + self.B1  # Nx2 * 2x2 -> Nx2
        self.H1 = self.sigmoid(self.A1)  # Nx2

        self.A2 = np.matmul(self.H1, self.W2) + self.B2  # Nx2 * 2x4 -> Nx4
        self.Y_hat = self.softmax(self.A2)  # Nx4
        return self.Y_hat

    def grad_sigmoid(self, X):
        return X * (1 - X)  # Nx2

    def grad(self, X, Y):
        self.forward_pass(X)
        # dl/da2 -> Nx4
        self.dA2 = self.Y_hat - Y
        # dl/da2 * da2/dw2 -> 2xN * Nx4
        self.dW2 = np.matmul(self.H1.T,
                             self.dA2)  # da2/dw2 = H1. 2xN * Nx4: 2x4, sum of gradients across all input example happens here because as N is inner
        self.dB2 = np.sum(self.dA2, axis=0).reshape(1, -1)
        # dl/da2 * da2*dh1
        self.dH1 = np.matmul(self.dA2, self.W2.T)  # da2*dh1 = W2: W2: 2x4: dA2: Nx4
        self.dA1 = self.grad_sigmoid(self.H1) * self.dH1  # dh1: Nx2, H1: Nx2
        self.dW1 = np.matmul(X.T,
                             self.dA1)  # Nx2 * Nx2 : 2x2, ssum of gradients across all input example happens here because as N is inner
        self.dB1 = self.dA1
        # since dA1 has shape Nx2 and dB1 has shape 1x2, we need to sum up the gradients across N
        self.dB1 = np.sum(self.dB1, axis=0).reshape(1, -1)

    def fit(self, X, Y, epochs=1, lr=1, display_loss=False):
        if display_loss:
            loss = {}
        for epoch in tqdm(range(epochs)):
            m = X.shape[0]
            self.grad(X, Y)
            self.W2 -= lr * self.dW2 / m
            self.W1 -= lr * self.dW1 / m
            self.B2 -= lr * self.dB2 / m
            self.B1 -= lr * self.dB1 / m
            if display_loss:
                Y_pred = self.predict(X)
                loss[epoch] = log_loss(np.argmax(Y, axis=1), Y_pred)
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.show()

    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred)

if __name__ == "__main__":
    model = FullVectorizedNN()
    data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
    labels_orig = labels
    labels = np.mod(labels_orig, 2)
    X_train, X_val, Y_train, Y_val = train_test_split(data, labels_orig, stratify=labels_orig, random_state=0)

    enc = OneHotEncoder()
    # 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
    y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
    y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()
    print(y_OH_train.shape, y_OH_val.shape)

    model.fit(X_train, y_OH_train, epochs=2000, lr=0.5, display_loss=True)

    Y_pred_train = model.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train, 1)

    Y_pred_val = model.predict(X_val)
    Y_pred_val = np.argmax(Y_pred_val, 1)

    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_val, Y_val)

    # print("Model {}".format(idx))
    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))
