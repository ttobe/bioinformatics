# 영상
# https://www.youtube.com/watch?v=IwLOWwrz26w&list=PL7ZVZgsnLwEEIC4-KQIchiPda_EjxX61r&index=4
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision import datasets
import torchvision

import numpy as np
import matplotlib.pyplot as plt

# 전처리 tensor로 바꿔주는 형태, normalize, 
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5), std = (1.0))])
                                        
# 데이터 로드                                       
trainset = datasets.MNIST(root="pytorch/", train=True, 
                            download=True, transform=mnist_transform)
testset = datasets.MNIST(root="pytorch/", train=False, 
                            download=True, transform=mnist_transform)
# 데이터 로더                             
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

dataiter = iter(train_loader)
images, labels = next(dataiter)

# batchsize, 1(흑백), 28, 28 / label은 batch_size
print(images.shape, labels.shape)

# 이미지 보여주기
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,5))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
# 4개면 그리드로 나타내기
# imshow(torchvision.utils.make_grid(images[:4]))

# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features

net = Net()
print(net)

# para
params = list(net.parameters())
print(len(params))

# 임의의 값 통과
input = torch.randn(1,1,28,28)
out = net(input)
print(out)

# 손실함수, 옵티마이저
criteration = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

# 모델 학습
# total_batch = len(train_loader)
# print(total_batch)
# # 469

# for epoch in range(10):

#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criteration(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 100 == 99:
#             print("epoch: {}, iter: {}, loss: {}".format(epoch+1, i+1, running_loss/2000))
#             running_loss = 0.0

# 저장하기        
PATH = 'pytorch/model.pth'
# torch.save(net.state_dict(), PATH)

# 모델 불러오기
net = Net()
net.load_state_dict(torch.load(PATH))

dataiter = iter(test_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:4]))
outputs = net(images)
_, predicted = torch.max(outputs,1)
print(''.join('{}\t'.format(str(predicted[j].numpy())) for j in range(4)))


correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(100 * correct / total)
