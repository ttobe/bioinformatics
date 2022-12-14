import numpy as np
from Codes.Activation_func import ReLu
from collections import OrderedDict
from Codes.SoftmaxWithLoss import SoftmaxWithLoss

# Affine 계층 Class
class Affine:
    # 가중치 값(W), 편향(b) 선언
    # 해당 Affine 계층으로 입력되는 inpur(x), W의 미분값(dw), b의 미분값(db)선언
    def __init__(self, Weight, Bias):
        self.W, self.b = Weight, Bias
        self.x, self.dW, self.db = None, None, None
    
    # Affine 계층의 순전파 함수
    # train_flg = 학습인지 여부(Drop Out Layer의 사용을 위한 변수)
    # input과, 가중치 값을 dot product한 후 편향을 더해 출력
    def forward(self, x, train_flg=True):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    # Affine 계층의 역전파 함수
    # 역전파의 입력인 dout와 가중치(W)의 전치행렬과 dot product후 input(x)의 미분 값 반환
    # Affine 층의 가중치 기울기, 편향 기울기를 계산하여 저장
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

# 배치 정규화 계층 Class    
class BN:
    # gamma, beta, momentum 초기화
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # test시  사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    # 배치 정규화 순전파 함수
    # train_flg = 학습인지 여부(Drop Out Layer의 사용을 위한 변수)
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        # 처음 이라면
        if self.running_mean is None:
            _, D = x.shape
            # 평균, 분산 저장할 변수 선언
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # 만일 training 중이라면            
        if train_flg:
            # mu(뮤): 배치의 평균
            mu = x.mean(axis=0)
            xc = x - mu
            # var: 분산
            var = np.mean(xc**2, axis=0)
            # std: 표준편차
            std = np.sqrt(var + 10e-7)
            # normalize
            xn = xc / std
            
            # 값 저장
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            # training의 평균, 분산 momentum을 기준으로 update 
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var 

        # test 중이라면
        else:
            # 그동한 구한 평균, 분산의 평균을 사용
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
        
        # xn에 gamma 곱하고 beta더하면 batch normaliztion 끝
        out = self.gamma * xn + self.beta 
        return out

    # 배치 정규화 역전파 함수
    def backward(self, dout):
        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        # dbeta:    beta 미분값
        # dgamma:   gamma 미분값
        # dxn:      xc / std 즉, normailaztion 미분값
        # dxc:      x - mu의 미분값
        # dstd:     표준편차의 미분값
        # dvar:     분산의 미분값
        # dxc:      x-mu의 미분값
        # dmu:      평균의 미분값
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        
        # batch_size로 나누어 반환
        dx = dxc - dmu / self.batch_size
        
        # dgamma, dbeta값은 학습을 위해 저장
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx

# Dropout Class
class Dropout:
    # drop out rate를 선언 및 초기화
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    # Drop out layer의 순전파 함수
    def forward(self, x, train_flg=True):
        # 만일 학습 중이라면, Drop Out사용
        if train_flg:
            # 랜덤으로 마스크를 생성 후 node를 비활성화 시킨 후 반환
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        # evaluate중이라면 Drop out 미사용
        # 단, 모든 노드에 sclaing을 위해 Dropout rate사용
        # 즉, 같은 출력값을 비교할 때 학습 시 적은 뉴런을 활용했을 때(상대적으로 많은 뉴런이 off 된 경우)와 
        # 여러 뉴런을 활용했을 때와 같은 scale을 갖도록 보정
        else:
            return x * (1.0 - self.dropout_ratio)
    
    # drop out layer의 역전파 함수
    # 기존 순전파 시 off된 node는 역전파 시 off되어야 하므로 mask적용
    def backward(self, dout):
        return dout * self.mask

# 네트워크(신경망) Class
class MultiLayer:
    # network의 input, output의 크기, 은닉층의 크기와, 가중치 초기화 표준편차, 배치 정규화, dropout 여부 등을 입력으로 받아 초기화
    # 
    def __init__(self, input_size, output_size, hidden_size_list, weight_init_std=0.01, use_bn=False, use_do=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        # hidden_layer_num: 은닉층 개수
        self.hidden_layer_num = len(hidden_size_list)
        # network에 필요한 모든 변수를 저장하는 dictionary변수
        self.params = {}
        self.__init_weight(weight_init_std)
        self.use_bn = use_bn
        self.use_do = use_do

        # layer변수는 순전파 진행을 위해 계층을 순서대로 dictionary 형태로 저장 
        self.layers = OrderedDict()
        # 은닉층의 수 만큼 반복하여 layer에 저장
        # 순서대로 Affine => (배치 정규화 사용 시) BatchNormalization => Activation Function 순으로 저장한다
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            # BN 사용
            if use_bn:
                # 배치 정규화 시 필요한 변수 gamma, beta를 은닉층의 번호에 맞게 저장한다
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BN' + str(idx)] = BN(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
            self.layers['Relu' + str(idx)] = ReLu()

            # drop out 사용
            if use_do:
                self.layers['Dropout' + str(idx)] = Dropout()

        # 마지막 softmax를 위한 게층 추가
        # Affine 게층 추가 후 activation function이 아닌 softmax 계층을 추가하여 마무리 한다
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    
    # 가중치 초기화 함수
    def __init_weight(self, weight_init_std):
        # input, output을 포함한 전체 network의 node수에 대한 list
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        # layer의 수만큼 반복하여 가중치(표준편차 사용), 편향 초기화
        for idx in range(1, len(all_size_list)):
            self.params['W'+str(idx)] = weight_init_std*np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])
    
    # network의 순전파 함수
    def predict(self, x, train_flg=True):
        # 신경망 통과
        # network에 저장된 모든 layer의 순전파를 진행하며, 이전 layer의 출력을 다음 layer의 입력으로 사용
        for layer in self.layers.values():
            x = layer.forward(x, train_flg)
        # 최종 출력 값을 반환
        return x

    # network loss를 계산 함수
    def loss(self, x, t, train_flg=False):
        # last_layer인 softmax layer직전까지 신경망을 통과
        y = self.predict(x, train_flg)
        # last_layer: Softmax_with_loss이므로 마지막 layer의 출력값(softmax의 loss)을 반환
        return self.last_layer.forward(y, t)
    
    # network의 정확도 계산 함수
    def accuracy(self, x, t, train_flg=True):
        # 예측값은 정규화 하지 않아도 결과는 변함 없기에 softmax layer는 생략
        y = self.predict(x, train_flg)
        # 출력값 중 최대값을 가지는 index(label)의 값을 예측값으로 선정
        y = np.argmax(y, axis=1)
        # 정답 data가 1차원이 아니라면, axis=1로 하여 정답 data추출
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        # 정확도 계산(batch size를 고려하여, 평균을 산출한다)
        acc = np.sum(y==t)/float(x.shape[0])
        return acc
    
    # network에 저장되어있는 변수(ex. W1, W2, etc.)의 기울기(미분 값)계산
    # main 함수에서는 gradient 함수만 호출
    def gradient(self, x, t, train_flg=True):
        # loss함수를 호출함으로 network에 변수들 저장
        self.loss(x, t, train_flg=True)
        dout = 1
        # last_layer인 softmax layer는 layer dictionary에 포함 되어있지 않으므로 개별 호출
        dout = self.last_layer.backward(dout)
        # 역전파를 위해 network layer를 reverse하여 저장
        layers = list(self.layers.values())
        layers.reverse()
        # 역전파 진행
        # 이전 layer의 출력을 다음 layer의 입력으로 사용
        for layer in layers:
            dout = layer.backward(dout)
        
        # 역전파 진행이 끝났다면, 모든 layer에 각 변수에 대한 미분 값이 저장되어있으므로
        # 각 layer의 기울기를 저장(grad)
        grads = {}
        # last_layer의 존재 및 0번이 아닌 1번 layer부터 존재하므로 hidden_layer_num+2까지 진행
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
            # 배치 정규화를 사용한다면 gamma, beta 변수의 미분 값 또한 저장
            # 하지만 last_layer에는 배치 정규화를 사용하지 않으므로, pass
            if idx != self.hidden_layer_num+1 and self.use_bn:
                grads['gamma' + str(idx)] = self.layers['BN' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BN' + str(idx)].dbeta

        # 저장한 기울기 값들 반환
        return grads