import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plot

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        num_layers = 4
        feature_out = [6, 16]
        # 32-> 28> 14> 10> 5
        last_out_dim = 5
        last_conv_feat_out = feature_out[-1] * last_out_dim * last_out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, feature_out[0], 5),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(feature_out[0], feature_out[1], 5),
            nn.AvgPool2d(2, 2)
        )
        fc_out = [120, 84, 10]
        self.fc = nn.Sequential(
            nn.Linear(last_conv_feat_out, fc_out[0]),
            nn.Linear(fc_out[0], fc_out[1]),
            nn.Linear(fc_out[1], fc_out[2]),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


def train(model, epochs, device):
    train_loader, test_loader = create_dataset()
    train_iter = iter(train_loader)
    loss_fn = nn.CrossEntropyLoss().to(device)
    opt = optim.Adam(model.parameters())

    loss_arr = []
    loss_epoch_array = []
    for epoch in (range(epochs)):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outs = model(inputs)
            loss = loss_fn(outs, labels)
            loss.backward()
            opt.step()
            loss_arr.append(loss.item())
    loss_epoch_array.append(loss.item())
    print(
        f"Epoch: {epoch}, loss: {loss.item()}, train_accuracy = {evaluation(train_loader.to(device), model)}, test_accuracy: {evaluation(test_loader.to(device), model)}")
    plt.plot(loss_epoch_array)
    plt.show()


def evaluation(loader, model):
    total = 0
    correct = 0
    for data in loader:
        inputs, labels = data
        out = model(inputs)
        _, pred = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    return 100 * correct / total


def create_dataset():
    trainset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, download=True,
                                            transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, download=True,
                                           transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)
    return trainloader, testloader


if __name__ == '__main__':
    model = LeNet()
    # train(model, 16)
    device = 'cpu' #"mps" if torch.backends.mps.is_available() else "cpu"
    # device = torch.device(device)
    print(f"Using device: {device}")
    model.to(device)
    train(model, 16, device)
