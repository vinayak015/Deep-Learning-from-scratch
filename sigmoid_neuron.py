import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score, mean_squared_error
from utils import prepare_mobile_dataset
from tqdm import tqdm


class SigmoidNeuron:

    def __init__(self):
        self.w = None
        self.b = None

    def perceptron(self, x):
        return np.dot(x, self.w.T) + self.b

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def grad_w(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def grad_b(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)

    def fit(self, X, Y, epochs=1, learning_rate=1, init=True, display_loss=False):
        if init:
            self.w = np.random.randn(1, X.shape[1])
            self.b = 0
        if display_loss:
            loss = {}
        for epoch in tqdm(range(epochs)):
            grad_w, grad_b = 0, 0
            for x, y in zip(X, Y):
                grad_w = grad_w + self.grad_w(x, y)
                grad_b = grad_b + self.grad_b(x, y)
            # update is happening at each epoch
            self.w = self.w - learning_rate * grad_w
            self.b = self.b - learning_rate * grad_b

            if display_loss:
                Y_pred = self.sigmoid(self.perceptron(X))
                loss[epoch] = mean_squared_error(Y_pred, Y)
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.sigmoid(self.perceptron(x))
            Y_pred.append(y_pred)
        return np.array(Y_pred)

""""

Uncomment this to train on toy dataset

X = np.asarray([[2.5, 2.5], [4, -1], [1, -4], [-3, 1.25], [-2, -4], [1, 5]])
Y = [1, 1, 1, 0, 0, 0]
sn = SigmoidNeuron()
epochs = 50
lr = .25
sn = SigmoidNeuron()
sn.fit(X, Y, 1, 0.25, True)

def plot_sn(X, Y, sn, ax):
  X1 = np.linspace(-10, 10, 100)
  X2 = np.linspace(-10, 10, 100)
  XX1, XX2 = np.meshgrid(X1, X2)
  YY = np.zeros(XX1.shape)
  for i in range(X2.size):
    for j in range(X1.size):
      val = np.asarray([X1[j], X2[i]])
      YY[i, j] = sn.sigmoid(sn.perceptron(val))
      my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
  ax.contourf(XX1, XX2, YY, cmap=my_cmap, alpha=0.6)
  ax.scatter(X[:,0], X[:,1],c=Y, cmap=my_cmap)
  ax.plot()

sn.fit(X, Y, 1, 0.05, True)
N = 30
plt.figure(figsize=(10, N*5))
for i in range(N):
  print(sn.w, sn.b)
  ax = plt.subplot(N, 1, i + 1)
  plot_sn(X, Y, sn, ax)
  sn.fit(X, Y, 1, 0.5, False)
"""

X_train, X_test, Y_train, Y_test, Y_train_binarised, Y_test_binarised = prepare_mobile_dataset()

sn2 = SigmoidNeuron()
sn2.fit(X_train, Y_train, epochs=2000, learning_rate=0.015, display_loss=True)



