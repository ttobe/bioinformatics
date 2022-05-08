import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
# RNN y = 0.5sini(x)-cos(x/2) 
# 시계열 데이터란 일정한 시간동안 수집 된 일련의 순차적으로 정해진 데이터 셋의 집합 입니다. 시계열 데이터의 특징으로는 시간에 관해 순서가 매겨져 있다
# 시계열 데이터를 이용해서 미래 값을 예측하는 RNN구조라면 일정한 길이의 패턴을 잘라서 학습 데이터를 만들어야 함
# 몇개를 묶을건지 window size w 설정
# 얼마나 먼 미래의 값을 예측할것인지 수평선 계수 horizon factor h 설정

# 입력데이터는 batch_size, time steps, input_dims 같은 구조로 해야함
# batch_size = window size로 분리되어 있는 데이터의 총 개수
# time steps = 몇개의 데이터를 이용해서 정답을 만들어 내는지 window size의 크기와 동일
# input_dims = RNN 레이어로 한번에 들어가는 데이터의 개수

# x = 0 ~ 100, unit-> 0.1 >>>> 1000개의 데이터
x = np.arange(0, 100, 0.1)
y = 0.5 * np.sin(2 * x) - np.cos(x / 2.0)

#for 3D tensor input, reshape(-1, 1) >> 단순히 1000개의 수가 아닌 (1000 X 1) 행렬로 바꿔줌
seq_data = y.reshape(-1, 1)
print(seq_data.shape)
print(seq_data[:5])

def seq2dataset(seq, window, horizon):
# X.shape를 batch size, time steps, input dims로 바꿔줌.....
    X = []
    Y = []

    for i in range(len(seq)-(window+horizon)+1):
        # window 20 단위로 슬라이싱
        x = seq[i:(i+window)]
        # windo 20 + horizion 1 - 1(index는 0부터) 해서 다음 정답 값을 Y에 넣기
        y = (seq[i+window+horizon-1])

        X.append(x)
        Y.append(y)

    #print(X[:5])
    #x가 2차원 행렬인데, np를 통해서 3차원 텐서로 변환되어 리턴 하나의 차원을 추가시켜서
    return np.array(X), np.array(Y)

w = 20 #window size
h = 1 #horizon factor 바로 다음 데이터 예측
X,Y = seq2dataset(seq_data,w,h)
print(X.shape, Y. shape)
#(980, 20, 1) (980, 1)

#검증을 위한 8:2 비율로 분리
split_ratio = 0.8

split = int(split_ratio*len(X))
x_train = X[0:split]
y_train = Y[0:split]
x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Sequential()
#model.add(SimpleRNN(units=simpleRNN layer내의 노드 개수, activation='tanh',input_shape=(window size, RNN 레이어로 한번에 몇개의 데이터가 들어가는지)))
model.add(SimpleRNN(units=128, activation='tanh',input_shape=(20,1)))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mae는 오차의 절대값
hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

pred = model.predict(x_test)

print(pred.shape)


rand_idx = np.random.randint(0, len(y_test), size=5)

print('random idx = ',rand_idx, '\n')

print('pred  = ', pred.flatten()[rand_idx])
print('label = ', y_test.flatten()[rand_idx])

#print(pred[rand_idx])
#print(y_test[rand_idx])

plt.plot(pred, label='prediction')
plt.plot(y_test, label='label')
plt.grid()
plt.legend(loc='best')

plt.show()



#과거의 정보가 마지막까지 충분히 전달되지 못함 - 장기 의존성 문제
