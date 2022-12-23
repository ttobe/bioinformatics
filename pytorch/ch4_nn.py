import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000
num_epoch = 10000
# 1000, 1배열, 원소가 -10 ~ 10
x = init.uniform_(torch.Tensor(num_data,1), -15, 15)
# 현실성을 반영하기 위해 노이즈가 추가된 상태로 ....... 가우시안 노이즈
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)
y = x ** 2 + 3
y_noise = y + noise

model = nn.Sequential(
    nn.Linear(1, 6),
    nn.ReLU(),
    nn.Linear(6,10),
    nn.ReLU(),
    nn.Linear(10,6),
    nn.ReLU(),
    nn.Linear(6,1)
)

# 차이의 절대값 평균
loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr = 0.002)
loss_array = []

for i in range(num_epoch):
    optimizer.zero_grad()
     
    output = model(x)

    loss = loss_func(output, y_noise)
    loss.backward()
    optimizer.step()

    loss_array.append(loss.data)

import matplotlib.pyplot as plt
plt.plot(loss_array)
plt.show()