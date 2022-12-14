import torch
import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        c = np.max(x) # 최댓값
        exp_a = np.exp(x-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        self.out = y
        return y
        # if -x > np.log(np.finfo(type(x)).max):
        #     return 0.0
        # out = 1/(1+np.exp(-x))
        # self.out = out
        # return out
    def backward(self, dout):
        dx = dout*self.out*(1-self.out)
        return dx

class ReLu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x, train_flg=True):
        self.mask = (x>=0)
        # self.mask 에는 input으로 들어온 x에 대해 0보다 크면 TRUE, 0보다 작으면 FALSE가 되는 bool ndarray 저장
        out = x.copy()
        out[self.mask] = 0
        # mask 적용 후 반환
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        # ReLu의 back propagation은 0이상의 data는 그대로 다음 layer로 보내줌
        return dx