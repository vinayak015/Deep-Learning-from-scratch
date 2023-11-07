import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm_notebook
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs


my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])

class MultiClassNN:
    """
    This class is creating a NN with 1 hidden layer and 1 output layer.
    The hidden layer has 2 neurons and the output layer has 4 neurons.
    The NN implemented is shown above
    """

    def __init__(self):
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.w7 = np.random.randn()
        self.w8 = np.random.randn()
        self.w9 = np.random.randn()
        self.w10 = np.random.randn()
        self.w11 = np.random.randn()
        self.w12 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward_pass(self, x):
        self.x1, self.x2 = x

        # Hidden layer
        self.a1 = self.w1 * self.x1 + self.w2 * self.x2
        self.h1 = self.sigmoid(self.a1)
        self.a2 = self.w3 * self.x1 + self.w4 * self.x2
        self.h2 = self.sigmoid(self.a2)

        # Output layer
        self.a3 = self.w5 * self.h1 + self.w6 * self.h2 + self.b3
        self.a4 = self.w7 * self.h1 + self.w8 * self.h2 + self.b4
        self.a5 = self.w9 * self.h1 + self.w10 * self.h2 + self.b5
        self.a6 = self.w11 * self.h1 + self.w12 * self.h2 + self.b6
        sum_exps = np.sum([np.exp(self.a3), np.exp(self.a4), np.exp(self.a5), np.exp(self.a6)])
        self.h3 = np.exp(self.a3) / sum_exps
        self.h4 = np.exp(self.a4) / sum_exps
        self.h5 = np.exp(self.a5) / sum_exps
        self.h6 = np.exp(self.a6) / sum_exps

        return np.array([self.h3, self.h4, self.h5, self.h6])

    def grad(self, x, y):
        self.forward_pass(x)
        self.y1, self.y2, self.y3, self.y4 = y
        x1, x2 = x
        # the dw's connected to output layers will just be -2(y-y_hat), we can ignore 2 and say y_hat -y
        self.dw5 = (self.h3 - self.y1) * self.h3 * (1 - self.h3) * self.h1
        self.dw6 = (self.h3 - self.y1) * self.h3 * (1 - self.h3) * self.h2
        self.db3 = (self.h3 - self.y1) * self.h3 * (1 - self.h3)

        self.dw7 = (self.h4 - self.y2) * self.h4 * (1 - self.h4) * self.h1
        self.dw8 = (self.h4 - self.y2) * self.h4 * (1 - self.h4) * self.h2
        self.db4 = (self.h4 - self.y3) * self.h4 * (1 - self.h4)

        self.dw9 = (self.h5 - self.y3) * self.h5 * (1 - self.h5) * self.h1
        self.dw10 = (self.h5 - self.y3) * self.h5 * (1 - self.h5) * self.h2
        self.db5 = (self.h5 - self.y3) * self.h5 * (1 - self.h5)

        self.dw11 = (self.h6 - self.y4) * self.h6 * (1 - self.h6) * self.h1
        self.dw12 = (self.h6 - self.y4) * self.h6 * (1 - self.h6) * self.h2
        self.db6 = (self.h6 - self.y4) * self.h6 * (1 - self.h6)

        # dh1/da1 da1/dw1 , dh2/da2 da2/dw2
        self.dh1_dw1 = self.h1 * (1 - self.h1) * x1
        self.dh1_dw2 = self.h1 * (1 - self.h1) * x2
        self.dh2_dw3 = self.h2 * (1 - self.h2) * x1
        self.dh2_dw4 = self.h2 * (1 - self.h2) * x2

        # da/dh
        self.da3_dh1 = self.w5
        self.da4_dh1 = self.w7
        self.da5_dh1 = self.w9
        self.da6_dh1 = self.w11

        self.da3_dh2 = self.w6
        self.da4_dh2 = self.w8
        self.da5_dh2 = self.w10
        self.da6_dh2 = self.w12
        # dh/da
        self.dh3_da3 = self.h3 * (1 - self.h3)
        self.dh4_da4 = self.h4 * (1 - self.h4)
        self.dh5_da5 = self.h5 * (1 - self.h5)
        self.dh6_da6 = self.h6 * (1 - self.h6)

        # dL/dh
        self.dl_dh3 = (self.h3 - self.y1)
        self.dl_dh4 = (self.h4 - self.y2)
        self.dl_dh5 = (self.h5 - self.y3)
        self.dl_dh6 = (self.h6 - self.y4)

        # dl/dw
        self.dw1 = ((self.dl_dh3 * self.dh3_da3 * self.da3_dh1) + (self.dl_dh4 * self.dh4_da4 * self.da4_dh1) + (
                    self.dl_dh5 * self.dh5_da5 * self.da5_dh1) + (
                                self.dl_dh6 * self.dh6_da6 * self.da6_dh1)) * self.dh1_dw1

        self.dw2 = ((self.dl_dh3 * self.dh3_da3 * self.da3_dh1) + (self.dl_dh4 * self.dh4_da4 * self.da4_dh1) + (
                    self.dl_dh5 * self.dh5_da5 * self.da5_dh1) + (
                                self.dl_dh6 * self.dh6_da6 * self.da6_dh1)) * self.dh1_dw2

        self.dw3 = ((self.dl_dh3 * self.dh3_da3 * self.da3_dh2) + (self.dl_dh4 * self.dh4_da4 * self.da4_dh2) + (
                    self.dl_dh5 * self.dh5_da5 * self.da5_dh2) + (
                                self.dl_dh6 * self.dh6_da6 * self.da6_dh2)) * self.dh2_dw3

        self.dw4 = ((self.dl_dh3 * self.dh3_da3 * self.da3_dh2) + (self.dl_dh4 * self.dh4_da4 * self.da4_dh2) + (
                    self.dl_dh5 * self.dh5_da5 * self.da5_dh2) + (
                                self.dl_dh6 * self.dh6_da6 * self.da6_dh2)) * self.dh2_dw4

    def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False, display_weight=False):

        if display_loss:
            loss = {}

        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, dw9, dw10, dw11, dw12, db1, db2, db3, db4, db5, db6 = [0] * 18
            for x, y in zip(X, Y):
                self.grad(x, y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                dw7 += self.dw7
                dw8 += self.dw8
                dw9 += self.dw9
                dw10 += self.dw10
                dw11 += self.dw11
                dw12 += self.dw12
                db3 += self.db3
                db4 += self.db4
                db5 += self.db5
                db6 += self.db6

            m = X.shape[0]
            self.w1 -= learning_rate * dw1 / m
            self.w2 -= learning_rate * dw2 / m
            self.w3 -= learning_rate * dw3 / m
            self.w4 -= learning_rate * dw4 / m
            self.w5 -= learning_rate * dw5 / m
            self.w6 -= learning_rate * dw6 / m
            self.w7 -= learning_rate * dw7 / m
            self.w8 -= learning_rate * dw8 / m
            self.w9 -= learning_rate * dw9 / m
            self.w10 -= learning_rate * dw10 / m
            self.w11 -= learning_rate * dw11 / m
            self.w12 -= learning_rate * dw12 / m
            self.b3 -= learning_rate * db3 / m
            self.b4 -= learning_rate * db4 / m
            self.b5 -= learning_rate * db5 / m
            self.b6 -= learning_rate * db6 / m

            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)

            # if display_weight:
            #   weight_matrix = np.array([[self.b3, self.w5, self.w6,
            #                              self.b4, self.w7, self.w8,
            #                              self.b5, self.w9, self.w10,
            #                              self.b6, self.w11, self.w12],
            #                             [0, 0, 0,
            #                              self.b1, self.w1, self.w2,
            #                              self.b2, self.w3, self.w4,
            #                              0, 0, 0]])
            #   weight_matrices.append(weight_matrix)

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
        return np.array(Y_pred)


multi = MultiClassNN()

data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
labels_orig = labels
labels = np.mod(labels_orig, 2)
X_train, X_val, Y_train, Y_val = train_test_split(data, labels_orig, stratify=labels_orig, random_state=0)

enc = OneHotEncoder()
# 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
y_OH_train = enc.fit_transform(np.expand_dims(Y_train,1)).toarray()
y_OH_val = enc.fit_transform(np.expand_dims(Y_val,1)).toarray()
print(y_OH_train.shape, y_OH_val.shape)

multi.fit(X_train,y_OH_train,epochs=400,learning_rate=1,display_loss=True, display_weight=True)
Y_pred_train = multi.predict(X_train)
Y_pred_train = np.argmax(Y_pred_train,1)

Y_pred_val = multi.predict(X_val)
Y_pred_val = np.argmax(Y_pred_val,1)

accuracy_train = accuracy_score(Y_pred_train, Y_train)
accuracy_val = accuracy_score(Y_pred_val, Y_val)

print("Training accuracy", round(accuracy_train, 2))
print("Validation accuracy", round(accuracy_val, 2))