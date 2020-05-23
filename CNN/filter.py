import torch
import torch.nn as nn
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np

china = load_sample_image('china.jpg')/255.0
flower = load_sample_image('flower.jpg')/255.0
images = np.array([china, flower])
plt.imshow(images[0], cmap='gray')
plt.savefig('test.png')

images = torch.from_numpy(images)


images = images.reshape(images.shape[0], -1, images.shape[1], images.shape[2])
print(images[0].shape)


conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
new_img = conv2d(images.float())
print(f'Shape of new Image:{new_img.shape}')
plt.imshow(new_img[0].reshape(new_img.shape[2], new_img.shape[3],-1).detach().numpy())
plt.savefig('china_filt.png')

plt.imshow(new_img[1].reshape(new_img.shape[2], new_img.shape[3],-1).detach().numpy())
plt.savefig('flower_filt.png')


