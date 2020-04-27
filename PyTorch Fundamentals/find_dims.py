import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

train_set = torchvision.datasets.FashionMNIST(
root = './data/FashionMNIST2', #location on disk
train = True,
download = True,
transform = transforms.Compose([
    transforms.ToTensor()
])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

sample = next(iter(train_set))

image, label = sample

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) #in_channels = 1 becoz it is grayscale
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
#         print(self.conv2.shape) #Formula = O= ((n-f+2p)/s)+1 #https://www.deeplizard.com
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        # (1) input layer
        t = t
        
        #(2) hidden conv layer
        t = self.conv1(t) #convolution operation
        t = F.relu(t) #operation
        t = F.max_pool2d(t, kernel_size=2, stride=2) #operation
        
        #(3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t) #becuase they don't have learnable parameters, so they come from Functional module
        t = F.max_pool2d(t, kernel_size=2, stride=2) #operations
        
        #(4) Linear
        t = t.reshape(-1, 12*4*4) # 12 is num of output channels, 4 X 4 is height and width
        t = self.fc1(t)
        t = F.relu(t)
        
        #(5) Linear 2
        t = self.fc2(t)
        t = F.relu(t)
        
        # Output Layer
        t = self.out(t)
#         t = F.softmax(t, dim = 1) # crossentropy loss function implicitly performs the softmax operation
        
        return t
        

network = Network()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.01)


EPOCHS = 5

for epoch in range(EPOCHS):

    total_correct = 0
    total_loss = 0
    passes = 0

    for batch in train_loader:
        passes+=1
        images, labels = batch
        
        preds = network(images)
        loss = criterion(preds, labels)
        
        
        
        optimizer.zero_grad() #zero the gradients, otherwise grads+=loss.backward()
        loss.backward() #calculate gradients
        optimizer.step() #update weights
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
        
    print(f'epoch:{epoch}, total_correct:{total_correct}, total_loss:{total_loss}, in {passes} passes')
