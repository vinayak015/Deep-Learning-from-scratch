"""
This file contains the code for the vanilla RNN model from scratch.
Ref: OneFourthLabs
"""

from io import open
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden):
        concat = torch.cat([input_, hidden], dim=1)
        hidden = self.i2h(concat)
        out = self.h2o(hidden)
        out = self.softmax(out)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden):
        hidden = torch.sigmoid(self.i2h(input_) + self.h2h(hidden))
        out = self.h2o(hidden)
        out = self.softmax(out)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def create_dataset(filename="name2lang.txt"):
    languages = []
    data = []
    X = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            name, lang = line.split(',')
            name, lang = name.strip(), lang.strip()
            if not lang in languages:
                languages.append(lang)
            X.append(name)
            y.append(lang)
            data.append((name, lang))
    return X, y, languages, data, len(languages)

def all_letters():
    all_letters = string.ascii_letters + " .,;'"
    return all_letters


def one_hot_name_encoder(name):
    all_letters_ = all_letters()
    num_letters = len(all_letters_)

    one_hot = torch.zeros(len(name), 1, num_letters)
    for i, letter in enumerate(name):
        one_hot[i][0][all_letters_.find(letter)] = 1
    return one_hot


def infer(net, name):
    net.eval()
    name_ohe = one_hot_name_encoder(name)
    hidden = net.init_hidden()

    # iterate through length of name
    for i in range(name_ohe.size()[0]):
        output, hidden = net(name_ohe[i], hidden)
    return output

def dataloader(num_data_points, X_, y_, languages):
    idx = np.random.choice(len(X_), num_data_points, replace=False)
    data = []
    for i in range(len(idx)):
        name, lang = X_[i], y_[i]
        data.append((name, lang, one_hot_name_encoder(name), one_hot_lang_encoder(lang, languages)))
    return data


def eval(net, data_points, k, X_, y_, languages):
    data = dataloader(data_points, X_, y_, languages)
    correct = 0

    for name, lang, name_ohe, lang_ohe in data:
        output = infer(net, name)
        val, indices = output.topk(k)
        if lang_ohe in indices:
            correct += 1
    return correct / data_points


def train(net, opt, criterion, data_points, X, y, languages):
    opt.zero_grad()
    total_loss = 0

    data = dataloader(data_points, X, y, languages)
    for name, lang, name_ohe, lang_ohe in data:
        hidden = net.init_hidden()
        for i in range(name_ohe.size()[0]):
            output, hidden = net(name_ohe[i], hidden)
        loss = criterion(output, lang_ohe)
        # loss.backward()
        loss.backward(retain_graph=True)

        total_loss += loss
    opt.step()
    return total_loss / data_points

def one_hot_lang_encoder(lang, languages):
    return torch.tensor([languages.index(lang)], dtype=torch.long)


def train_setup(lr=0.01, n_batches=100, batch_size=10, momentum=0.9, display_freq=5):

    X, y, languages, data, num_languages = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    net = RNN2(len(all_letters()), 128, num_languages)
    criterion = nn.NLLLoss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    loss_arr = np.zeros(n_batches)

    for i in range(n_batches):
        loss = train(net, opt, criterion, batch_size, X_train, y_train, languages)
        loss_arr[i] = loss

        if i % display_freq == display_freq - 1:
            clear_output(wait=True)
            print('Iteration', i, 'Top-1:', eval(net, len(X_test), 1, X_test, y_test, languages), 'Top-2:',
                  eval(net, len(X_test), 2, X_test, y_test, languages), 'Loss', loss_arr[i])
            plt.figure()
            plt.plot(loss_arr, '-*')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()
            print('\n\n')

if __name__ == '__main__':
    train_setup(lr=0.0005, n_batches=100, batch_size=256)