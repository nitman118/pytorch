import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sklearn import datasets

#prepare datasets

X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise=20, random_state = 1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

print(f'Shape of X:{X.shape}')
print(f'Shape of y:{y.shape}')

y = y.view(y.shape[0],1)

#model
n_samples, n_features = X.shape
input_size = n_features
output_size=1

class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self)
