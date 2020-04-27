# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# print(torch.cuda.current_device())
print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

train_set = torchvision.datasets.MNIST(
root = './data/MNIST', #location on disk
train = True,
download = True,
transform = transforms.Compose([
    transforms.ToTensor()
])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

image, label  = next(iter(train_set))

print(f'Shape of image:{image.shape}')

images, labels = next(iter(train_loader))

print(f'Shape of batch of images:{images.shape}')

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=256)
        self.batch_norm1 = nn.BatchNorm1d(num_features=256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.batch_norm2 = nn.BatchNorm1d(num_features=128)
        self.dropout2 = nn.Dropout(p=0.2)
        self.output = nn.Linear(in_features=128, out_features=10)

    def forward(self, t):
        #input
        t = t
        t = t.reshape(-1,1*28*28)
        #Input Group 1
        t = F.relu(self.fc1(t))
        t = self.batch_norm1(t)
        t = self.dropout1(t)
        #Input Group 2
        t = F.relu(self.fc2(t))
        t = self.batch_norm2(t)
        t = self.dropout2(t)
        #Output
        t = self.output(t)

        return t


net = Network()

# pred = net(image.unsqueeze(0)) 
pred = net(images)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.01)

NUM_EPOCHS = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

net = net.train()
for epoch in range(NUM_EPOCHS):

    loss_stat = 0

    for batch in train_loader:
        images, labels = batch[0].to(device), batch[1].to(device)

        preds = net(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_stat += loss.item()

    print(f'Epoch:{epoch}, loss:{loss_stat}')

net = net.eval()

def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds



with torch.no_grad(): #you could also decorate the function @torch.no_grad()
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = 10000)
    train_preds = get_all_preds(net, prediction_loader)

stacked = torch.stack([train_set.targets, train_preds.argmax(dim=1)], dim=1)#dim = 1 vertical stacking

print(f'Accuracy:{get_num_correct(train_preds, train_set.targets)/len(train_set)}')

cmt = torch.zeros(10,10, dtype = torch.int32)
for p in stacked:
    tl, pl = p.tolist() #tl - true label; pl- predicted label
    cmt[tl,pl]+=1

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
names = [str(i) for i in range(10)]
df_cm = pd.DataFrame(cm, index = names,
                  columns = names)

plt.figure(figsize = (10,8))
sn.heatmap(df_cm, annot=True, square = True, fmt='g',cmap="YlGnBu");  #https://seaborn.pydata.org/generated/seaborn.heatmap.html
plt.xlabel('Predicted');
plt.ylabel('Actual')
plt.savefig('cm.png')




