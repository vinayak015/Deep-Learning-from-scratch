"""
Moving from python, numpy to pytorch.
from course by One Fourth Labs
"""

import math
import matplotlib. pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

def create_dataset():
    data, labels = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
    plt.show()

    X_train, X_val, Y_train, Y_val =train_test_split(data, labels, stratify=labels, random_state=0)
    X_train, X_val, Y_train, Y_val = map(torch.tensor, (X_train, X_val, Y_train, Y_val))
    X_train, X_val, Y_train, Y_val = X_train.float(), X_val.float(), Y_train.long(), Y_val.long()
    return X_train, X_val, Y_train, Y_val

def model(x, weights1=None,  bias1=None, weights2=None, bias2=None):
    a1 = torch.matmul(x, weights1) + bias1
    h1 = a1.sigmoid()
    a2 = torch.matmul(h1, weights2) + bias2
    h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)
    return h2

def loss_fn(y_hat, y):
    return -(y_hat[range(y_hat.shape[0]), y].log()).mean()

def accuracy(y_hat, y):
    pred = torch.argmax(y_hat, dim=-1)
    return (pred==y).float().mean()


def fit_basic_model(train_data):
    weights1 = torch.randn(2, 2) / math.sqrt(2)
    weights1.requires_grad_()
    bias1 = torch.zeros(2, requires_grad=True)
    weights2 = torch.randn(2, 4) / math.sqrt(2)
    weights2.requires_grad_()
    bias2 = torch.zeros(4, requires_grad=True)
    lr = 0.2
    epochs = 1000
    X_train, X_val= train_data
    loss_arr = []
    acc_arr = []
    for epoch in tqdm(range(epochs)):
        y_hat = model(X_train, weights1, bias1, weights2, bias2)
        loss = loss_fn(y_hat, Y_train)
        loss.backward()
        loss_arr.append(loss)
        acc_arr.append(accuracy(y_hat, Y_train))

        with torch.no_grad():
            weights2 -= lr * weights2.grad
            weights1 -= lr * weights1.grad
            bias2 -= lr * bias2.grad
            bias1 -= lr * bias1.grad

            weights2.grad.zero_()
            weights1.grad.zero_()
            bias2.grad.zero_()
            bias1.grad.zero_()
    plt.plot([l.detach().numpy() for l in loss_arr], 'r-')
    plt.plot([l.detach().numpy() for l in acc_arr], 'b-')
    plt.show()
    print(f'loss before training {loss_arr[0]}')
    print(f'loss after training {loss_arr[-1]}')


class FirstNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.weights1 = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))
        self.bias1 = nn.Parameter(torch.zeros(2))

        self.weights2 = nn.Parameter(torch.randn(2, 4) / math.sqrt(2))
        self.bias2 = nn.Parameter(torch.zeros(4))

    def forward(self, X):
        a1 = torch.matmul(X, self.weights1) + self.bias1
        h1 = a1.sigmoid()
        a2 = torch.matmul(h1, self.weights2) + self.bias2
        h2 = a2.exp() / a2.exp().sum(-1).unsqueeze(-1)
        return h2


class Linear_v1(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(2, 2)
        self.lin2 = nn.Linear(2, 4)

    def forward(self, X):
        a1 = self.lin1(X)
        h1 = a1.sigmoid()
        a2 = self.lin2(h1)
        h2 = a2.exp() / a2.exp().sum(-1).unsqueeze(-1)
        return h2


class Linear_v2(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 4),
            nn.Softmax()
        )

    def forward(self, X):
        return self.net(X)

def fit_v1(epochs=1, lr=1, model=None, train_data=None):
    loss_arr = []
    acc_arr = []
    for epoch in tqdm(range(epochs)):
        y_hat = model(X_train)
        loss = F.cross_entropy(y_hat, Y_train)
        loss.backward()
        loss_arr.append(loss.item())
        acc_arr.append(accuracy(y_hat, Y_train))

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
            model.zero_grad()
    plt.plot(loss_arr, 'r-')
    plt.plot([l.detach().numpy() for l in acc_arr], 'b-')
    plt.show()
    print(f'loss before training {loss_arr[0]}')
    print(f'loss after training {loss_arr[-1]}')



def fit_v2(epochs=1, lr=1, model=None, train_data=None):
    loss_arr = []
    acc_arr = []
    opt = optim.SGD(model.parameters(), lr)
    for epoch in tqdm(range(epochs)):
        y_hat = model(X_train)
        loss = F.cross_entropy(y_hat, Y_train)
        loss_arr.append(loss.item())
        acc_arr.append(accuracy(y_hat, Y_train))

        loss.backward()
        opt.step()
        opt.zero_grad()

    plt.plot(loss_arr, 'r-')
    plt.plot([l.detach().numpy() for l in acc_arr], 'b-')
    plt.show()
    print(f'loss before training {loss_arr[0]}')
    print(f'loss after training {loss_arr[-1]}')

def fit_v3(epochs=1, lr=1, model=None, train_data=None):
    opt = optim.SGD(model.parameters(), lr)
    X_train, X_val = train_data
    for epoch in tqdm(range(epochs)):
        loss = F.cross_entropy(model(X_train), Y_train)

        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item()


if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = create_dataset()
    train_data = X_train, Y_train
    fit_basic_model(train_data)
    fit_v1(epochs=1000, lr=0.2, model=FirstNetwork(), train_data=train_data)
    fit_v1(epochs=1000, lr=0.2, model=Linear_v1(), train_data=train_data)
    fit_v2(epochs=1000, lr=0.2, model=Linear_v1(), train_data=train_data)
    fit_v2(epochs=1000, lr=0.2, model=Linear_v2(), train_data=train_data)
    fit_v3(epochs=1000, lr=0.2, model=Linear_v2(), train_data=train_data)
