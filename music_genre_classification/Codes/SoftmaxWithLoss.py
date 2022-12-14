import torch
import numpy as np

def softmax(x):
    max = np.max(x, axis=1)
    # (batch_size, output_feature 수)의 행렬을 행 별로 최댓값 추출
    # max 의 shape = (batch_size, ) 1 dim tensor
    max = max.reshape(-1, 1)
    # 1dim tensor의 broadcasting을 위해 2dim tensor로 reshpe
    # why? (batch_size, output_feature 수)shape의 x가 2dim tenosor이므로 reshape 필요
    denominator = np.sum(np.exp(x-max), axis=1)
    denominator = denominator.reshape(-1, 1)
    # denominator(분모) 또한 행별로 sum을 한 후 2dim tensor와 broadcasting을 위해 reshape
    numerator = np.exp(x-max)
    # numerator(분자) 는 2dim인 x와 broadcasting으로 2dim tensor
    return numerator/denominator
    # 즉 행=batch, 열=output feature의 확률을 나타내는 2dim tensor반환
    # return shape = (batch_size, output_feature 수)

def cross_entropy_error(y, t):
    ##### batch 학습이 아닌 data 1개씩 학습 할 경우 실행 code
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    ##### batch 학습이 아닌 data 1개씩 학습 할 경우 실행 code
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size 
    
    # return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size
    # 위 반환식: one_hot_encoding(X) | labeling(O)

    # return -np.sum(t*np.log(y+1e-7))/batch_size 
    # 위 반환식: one_hot_encoding(O) | labeling(X)

    # scalar 값 1개(loss) batch 평균 loss반환
    # 1e-7은 log함수의 발산 방지용
    
# Softmax Class
class SoftmaxWithLoss:
    # softmaxlayer 통과 후 loss 계산 반환
    # loss 값, 예측값(y), 정답(t)선언
    def __init__(self):
        self.loss = None
        self.y, self.t = None, None
    # softmax layer의 순전파 함수
    # 인자로 전달받은 input(x)를 softmax함수를 이용해 예측값(y)를 반환받은 후 저장
    # 예측값(y)와 정답(t)를 cee를 사용하여 loss값을 저장 후 반환
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    # softmax layer의 역전파 함수
    # last_layer인 softmax layer이므로 dout의 default값은 1로 설정
    # softmax의 역전파는 y-t로 간단하게 나온다.
    # 따라서 배치크기로 나누어 평균 값을 반환한다
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx