import torch
x = torch.Tensor(2,3)
print(x)
# 텐서는 다차원 배열
x = torch.tensor([[1,2,3], [4,5,6]])
print(x)

x = torch.tensor(data=[2.0, 3.0], requires_grad=True)
y = x ** 2
z = 2 * y + 3
print(x, y, z)
target = torch.tensor([3.0, 4.0])
loss = torch.sum(torch.abs(z-target))
loss.backward()

print(x.grad, y.grad, z.grad)


import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000
num_epoch = 500
# 1000, 1배열, 원소가 -10 ~ 10
x = init.uniform_(torch.Tensor(num_data,1), -10, 10)
# 현실성을 반영하기 위해 노이즈가 추가된 상태로 ....... 가우시안 노이즈
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)
y = 2 * x + 3
y_noise = 2 * (x + noise) + 3

# 1개의 특성을 가진 1000개 데이터 -> 1, 1 out도 1이니까
model = nn.Linear(1,1)
# 차이의 절대값 평균
loss_func = nn.L1Loss()
# optimizer SGD를 사용하고, 최적화할 변수로 model의 파라미터를 넘겨주었다.
optimizer = optim.SGD(model.parameters(), lr = 0.01)
list_loss = []
label = y_noise
for i in range(num_epoch):
    # 지난번에 계산 했던 기울기를 0으로 초기화 시킴 
    # 그래야 새로운 가중치와 편차에 대해 새로운 기울기를 구할 수 있음
    optimizer.zero_grad()
    output = model(x)

    # loss 함수로 예측값과 정답의 차이를 저장
    loss = loss_func(output, label)
    list_loss.append(loss.data)
    # 기울기 계산 
    loss.backward()
    # 변수들의 기울기에 학습률 0.01을 곱해서 빼주면서 업데이트
    optimizer.step()

    if i % 10 == 0:
        print(loss.data)
        
import matplotlib.pyplot as plt
plt.plot(list_loss)
plt.show()
param_list = list(model.parameters())
print(param_list[0].item(), param_list[1].item())