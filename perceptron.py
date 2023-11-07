"""
Back to basics: Revisiting the perceptron model proposed by Frank Rosenblatt in 1957.
This is a binary classifier.
It is a simple model which takes in real inputs and gives a binary output.
Opportunity to code the perceptron model from scratch by One Fourth Labs.
"""


import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import prepare_breast_cancer_data as prepare_data


class Perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def model(self, x):
        return np.dot(self.w, x) >= self.b

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y, epochs=1, lr=1):
        self.w = np.ones(X.shape[1])
        self.b = 0
        accuracy = {}
        max_accuracy = 0
        for epoch in tqdm(range(epochs)):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w += lr * x
                    self.b += lr * 1
                if y == 0 and y_pred == 1:
                    self.w -= lr * x
                    self.b -= lr * 1
            accuracy[epoch] = accuracy_score(self.predict(X), Y)
            if accuracy[epoch] > max_accuracy:
                max_accuracy = accuracy[epoch]
                chkpt_w = self.w
                chkpt_b = self.b

        self.w = chkpt_w
        self.b = chkpt_b

        print('Max accuracy is ', max_accuracy)

        plt.plot(accuracy.values())
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.show()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_data()
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train, epochs=200, lr=0.001)

    Y_test_pred = perceptron.predict(x_test)
    accuracy_test = accuracy_score(Y_test_pred, y_test)
    print('Test accuracy is ', accuracy_test)
