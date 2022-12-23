import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5), std = (1.0))])
trainset = datasets.MNIST(root="pytorch/", train=True, 
                            download=True, transform=mnist_transform)
testset = datasets.MNIST(root="pytorch/", train=False, 
                            download=True, transform=mnist_transform)                            
train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)

dataiter = iter(train_loader)
images, labels = next(dataiter)
# batchsize, 1(흑백), 28, 28
print(images.shape, labels.shape)

torch_image = torch.squeeze(images[0])
print(torch_image.shape)

import matplotlib.pyplot as plt
print(torch_image)
plt.imshow(torch_image,cmap='gray')
# plt.show() 



