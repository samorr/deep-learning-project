# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import cifar10, mnist
from keras.utils import to_categorical
from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def getData():

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainset.data = trainset.data[:30000]
    trainset.targets = trainset.targets[:30000]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                            shuffle=True, num_workers=2)


    validset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    validset.data = validset.data[30000:]
    validset.targets = validset.targets[30000:]

    validloader = torch.utils.data.DataLoader(validset, batch_size=50,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                            shuffle=False, num_workers=2)
    return trainloader, validloader, testloader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.dropout(x, 0.2)
        x = x.view(-1, 16 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, 1)
        return x

def compute_accuracy(model, data_loader):
    model.eval()
    num_errs = 0.0
    num_examples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda()
            y = y.cuda()
            outputs = model.forward(x)
            _, predictions = outputs.data.max(dim=1)
            num_errs += (predictions == y.data).sum().item()
            num_examples += x.size(0)
    return 100.0 * num_errs / num_examples


def train(model, trainloader, optimizer, criterion, max_epoch,
            validloader=None, current_epoch=0, print_info=True):
    model = model.train()
    epoch = current_epoch
    while epoch <= max_epoch:

        if epoch - current_epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] /= 10.
                
        if epoch - current_epoch == 50:
            for g in optimizer.param_groups:
                g['lr'] /= 10.
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if print_info:
            print('[Epoch: %d] loss: %.3f, accuracy: %.2f, validation accuracy: %.2f' %
                (epoch + 1,
                loss,
                compute_accuracy(model, trainloader),
                compute_accuracy(model, validloader)),
                )
            model = model.train()
        epoch += 1

    print('Finished Training')

def save_model(model, filename):
    torch.save(model, filename)

if __name__ == '__main__':

    model = CNN()
    model = model.cuda()

    trainloader, validloader, testloader = getData()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train(model, trainloader, optimizer, criterion, 200, validloader)

    acc = compute_accuracy(model, testloader)

    save_model(model, 'model.pt')

    print('Accuracy of the network on the 10000 test images: %d %%' % acc)