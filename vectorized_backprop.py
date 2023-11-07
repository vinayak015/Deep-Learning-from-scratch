import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs


my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])


class VectorizedBackprop:
    """
    This class vectorizes weights and biases, not the input
    """

    def __init__(self):
        self.W1 = np.random.randn(2, 2)
        self.W2 = np.random.randn(2, 4)
        self.B1 = np.zeros((1, 2))
        self.B2 = np.zeros((1, 4))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, x):
        # x should be a vector
        exps = np.exp(x)
        return exps / np.sum(exps)

    def forward_pass(self, x):
        x = x.reshape(1, -1)  # (1,2) if x has 2 feat
        self.A1 = np.matmul(x, self.W1) + self.B1
        self.H1 = self.sigmoid(self.A1)  # 1x2

        self.A2 = np.matmul(self.H1, self.W2) + self.B2  # 1x4
        self.Y_hat = self.softmax(self.A2)
        return self.Y_hat

    def grad_sigmoid(self, x):
        return x * (1 - x)  # usually sigmoid(x) * (1-sigmoid(x))

    def grad(self, x, y):
        # dL/da2*dA2/dh1*dh1/da1* da1/dw1
        self.forward_pass(x)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)  # 1x4

        self.dA2 = self.Y_hat - y  # dL/da2 1x4
        self.dW2 = np.matmul(self.H1.T, self.dA2)  # dL/da2* da2/dw2 h1=1x2, da1=1x4
        self.dB2 = self.dA2
        self.dH1 = np.matmul(self.dA2, self.W2.T)  # dL/da2*da2/dh1 1x4 x 4x2 = 1x2
        self.dA1 = self.dH1 * self.grad_sigmoid(self.H1)  # #dL/da2 * dA2/dh1 * dh1/da1 -> 1x2
        self.dW1 = np.matmul(x.T, self.dA1)  # dL/da2*dA2/dh1*dh1/da1* da1/dw1 1x2,1x2
        self.dB1 = self.dA1

    def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False):
        if display_loss:
            loss = {}
        for i in tqdm(range(epochs), total=epochs, unit="epoch"):
            dW1 = np.zeros((2, 2))
            dW2 = np.zeros((2, 4))
            dB1 = np.zeros((1, 2))
            dB2 = np.zeros((1, 4))
            for x, y in zip(X, Y):
                self.grad(x, y)
                dW1 += self.dW1
                dW2 += self.dW2
                dB1 += self.dB1
                dB2 += self.dB2
            m = X.shape[0]  # no of samples
            # before updating the weights take the average of all the gradients to avoid oscillations as well as gradient explosion
            self.W1 -= learning_rate * dW1 / m  # dW1/m is the average of all the dW1s
            self.W2 -= learning_rate * dW2 / m  # dW2/m is the average of all the dW2s
            self.B1 -= learning_rate * dB1 / m  # dB1/m is the average of all the dB1s
            self.B2 -= learning_rate * dB2 / m  # dB2/m is the average of all the dB2s

            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.show()

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()


if __name__ == "__main__":
    model = VectorizedBackprop()
    data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
    labels_orig = labels
    labels = np.mod(labels_orig, 2)
    X_train, X_val, Y_train, Y_val = train_test_split(data, labels_orig, stratify=labels_orig, random_state=0)

    enc = OneHotEncoder()
    # 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
    y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
    y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()
    print(y_OH_train.shape, y_OH_val.shape)

    model.fit(X_train, y_OH_train, epochs=2000, learning_rate=0.5, display_loss=True)

    Y_pred_train = model.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train, 1)

    Y_pred_val = model.predict(X_val)
    Y_pred_val = np.argmax(Y_pred_val, 1)

    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_val, Y_val)

    # print("Model {}".format(idx))
    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))




