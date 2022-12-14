import torch
import numpy as np

class SGD:
    # 학습률 lr 초기화
    def __init__(self, lr = 0.01):
        self.lr = lr
    def step(self, params, grads):
        # grad와 학습률을 곱한 값을 빼주어서 파라미터 값을 조정
        for key in params.keys():
            params[key] -= self.lr*grads[key]

class Momentum:
    # 학습률 lr, Momentum 비율 초기화 및 이전의 값들을 저장할 리스트 v 선언
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def step(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
    # 현재의 학습률 변화에서 이전에 저장해 두었던 v리스트의 값 빼고, 그 값을 토대로 파라미터 갱신
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
    
    def step(self, params, grads): # h와 가중치 갱신
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # 기존 기울기 값을 제곱하여 계속 더해주기
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-7) # 가중치에 1/sqrt(h) 곱하여 갱신
            # 0으로 나누는 경우가 발생하지 않도록 1e-7을 더해준다.

class Adam:
    # beta1 = 일차 모멘텀용 계수, beta2 = 이차 모멘텀용 계수
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    
    def step(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        # 학습률 조정
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            # momentum 조정
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            # adagrad 조정
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7) # 0으로 나누는 경우가 발생하지 않도록 1e-7을 더해준다.
