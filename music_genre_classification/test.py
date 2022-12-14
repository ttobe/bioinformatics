import numpy as np
import Codes.Layer as Layer
import matplotlib.pyplot as plt
import Codes.optimizer as optimizer
from Data.data_preprocessing import data_preprocessing

### 신경망 생성 ###
# input_size:   feature 수
# input_size:   one-hot-encoding label 수
# hidden_size_list: 각 은닉층의 퍼셉트론(노드) 수
# weight_init_std:  가중치 초기화 표준편차
# use_bn:   배치 정규화 사용 여부
# use_do:   드롭 아웃 사용 여부
class Model:
    def __init__(self, params):
        ### data 불러오기 ###
        (self.x_train, self.t_train), (self.x_test, self.t_test) = data_preprocessing()
        ### 학습 변수 설정 ###
        # epochs_num:   학습 epoch 수
        # train_size:   학습 data의 개수
        # batch_size:   배치 크기
        # lr:           학습률
        # iter_per_epoch: epoch당 반복 횟수
        self.epochs_num = params['epoch']
        self.train_size = self.x_train.shape[0]
        self.batch_size = params['batch_size']
        self.iter_per_epoch = int(self.train_size / self.batch_size)
        self.lr = params['lr']
        h_list = []
        n_node = params['num_node']
        for i in range(params['num_hidden_layer']):
            h_list.append(n_node)
        
        self.network = Layer.MultiLayer(input_size=self.x_train.shape[1], 
                                output_size=self.t_train.shape[1], 
                                hidden_size_list=h_list, 
                                weight_init_std=0.01, 
                                use_bn=True, 
                                use_do=True)

        ### 결과 출력을 위한 list선언 ###
        # train_loss_list:  학습 시 loss값 저장 list
        # train_acc_list:   학습 시 정확도 저장 list
        # train_loss_list:  평가 시 loss값 저장 list
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        

        ### 초기 loss값 저장 ###
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)


        train_acc = self.network.accuracy(self.x_train, self.t_train, train_flg=False)
        test_acc = self.network.accuracy(self.x_test, self.t_test, train_flg=False)
        self.train_acc_list.append(train_acc)
        self.test_acc_list.append(test_acc)

    def train_network(self):
        
        ### optimizer 선언 ###
        # optimizer는 Adam으로 선언 및 학습률 전달
        
        optim = optimizer.Adam(self.lr)
        ### 학습 ###    
        for i in range(self.epochs_num):
            print("Epoch {}/{}".format(i+1, self.epochs_num))
            print("[", end='')
            for j in range(self.iter_per_epoch + 1):
                if j%3==0:
                    print(">", end='')
                batch_mask = np.random.choice(self.train_size, self.batch_size)
                x_batch = self.x_train[batch_mask]
                t_batch = self.t_train[batch_mask]
                # 신경망의 순전파, 역전파를 모두 진행하는 grad함수 호출하여 각layer의 기울기를 반환받는다
                grad = self.network.gradient(x_batch, t_batch, train_flg=True)
                # 반환된 grad 변수와 신경망의 param변수(W1, W2, b1, b2 etc...)들을 함께 optimizer에 전달하여 학습
                optim.step(self.network.params, grad)
            print(']')

            # 새로운 batch를 랜덤으로 선정
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            
            # 해당 data로 loss값 계산 및 저장
            loss = self.network.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)

            # test, training의 정확도를 계산 후  저장
            train_acc = self.network.accuracy(self.x_train, self.t_train, train_flg=False)
            test_acc = self.network.accuracy(self.x_test, self.t_test, train_flg=False)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            print("train acc: {} %".format(train_acc*100))
            print("test acc:  {} %".format(test_acc*100))
            print("loss:  {}".format(loss))
                
        plt.plot(self.train_acc_list, 'bo-')
        plt.xlim([0, 51])
        plt.plot(self.test_acc_list, 'm^-')
        plt.ylim([0,1])
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()

        plt.plot(self.train_loss_list, 'rx--')
        plt.show()