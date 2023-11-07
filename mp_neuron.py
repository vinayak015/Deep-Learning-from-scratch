"""
Back to basics: Revisiting the first neuron model proposed by Warren McCulloch and Walter Pitts in 1943.
This is McCulloch-Pitts Neuron Model, which is a binary classifier.
It is a simple model which takes in binary inputs and gives a binary output.
In this model, we have a parameter called b, which is a threshold value.
The model sums up all the inputs and if the sum is greater than the threshold value, it outputs 1, else 0.
Here we have used brute force to find the optimal value of b.
The inspiration is taken by Deep Learning Course by One Fourth Labs.
"""

from sklearn.metrics import accuracy_score
import numpy as np

from utils import prepare_breast_cancer_binarised_data as prepare_data


class MPNeuron:

    def __init__(self):
        self.b = None

    def model(self, x):
        return sum(x) >= self.b

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}

        for b in range(X.shape[1] + 1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y, Y_pred)

        best_b = max(accuracy, key=accuracy.get)
        self.b = best_b

        print('Opitmal Value of b is ', best_b)
        print('Highest accuracy is ', accuracy[best_b])


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_data()
    mp_neuron = MPNeuron()
    mp_neuron.fit(x_train, y_train)

    Y_test_pred = mp_neuron.predict(x_test)
    accuracy_test = accuracy_score(Y_test_pred, y_test)
    print('Test accuracy is ', accuracy_test)
