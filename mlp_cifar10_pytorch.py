# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import cifar10, mnist
from keras.utils import to_categorical
from torchvision import transforms
import torchvision
import torch

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

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.l1 = nn.Linear(3*32*32, 100)
        self.l2 = nn.Linear(100, 200)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.l1(x))
#         print(x.size())
        x = F.relu(self.l2(x))
#         print(x.size())
        x = F.relu(self.l3(x))
        x = F.dropout(x, 0.2)
#         print(x.size())
        x = self.l4(x)
        x = F.softmax(x, 1)
        return x

model = MLP()
# model.to_device('cuda')
model = model.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

model = model.train()

for epoch in range(100):
    
    if epoch == 10:
        for g in optimizer.param_groups:
            g['lr'] = 0.001
            
    if epoch == 50:
        for g in optimizer.param_groups:
            g['lr'] = 0.0001

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('[Epoch: %d] loss: %.3f, accuracy: %.2f, validation accuracy: %.2f' %
          (epoch + 1,
           loss,
           compute_accuracy(model, trainloader),
           compute_accuracy(model, validloader)),
         )
    model = model.train()
    

print('Finished Training')

acc = compute_accuracy(model, testloader)

print('Accuracy of the network on the 10000 test images: %d %%' % acc)