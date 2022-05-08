#RNN y = 0.5sini(x)-cos(x/2)
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM

#LSTM 시간 t에서 출력값 Ht이외에 LSTM레이어 사이에서 공유된느 Ct라는 셀상태 변수가 추가적으로 공유가 됨.
#기존의 상태를 보존함 -> 장기 의존성의 문제를 해결함

#핵심idea: 이전 단계 정보를 memory cell에 저장하여 다음 단계로 전달하는 것
#현재 시점의 정보를 바탕으로 과거 내용을 얼마나 잊을지 또는 기억할지 등을 계산하고,
#그 결과에 현재 정보를 추가해서 다음시점으로 정보를 전달함

#forget gate 과거의 정보를 얼마나 잊을지, 가져갈지
#현시점의 데이터와 과거의 은닉층 값에 각각의 가중치를 곱한 후에 더하고 sigmoid
#결과가 1에 가까울수록 과거의 정보를 많이 가져간다

#input gate 현재의 정보를 과거 정보에 얼마나 반영할지
#현재시점의 데이터와 과거의 은닉층 값에 각각의 가중치를 곱하고 더하고 sigmoid
#어떤 정보를 업데이트 할 지 결정하는 것
#현재 시점의 데이터와 과거 은닉층의 값에 가중치를 곱하고 더하고 tanh
#현재시점의 새로운 정보를 생성함
#현재시점에서 실제로 갖고있는 정보가 얼마나 중요한지 반영하여 cell에 기록

#cell state forgate gate출력값 F, input gate 출력값을 이용해 memory cell에 저장하는 단계
#과거의 정보를 forget gate에서 계산된 만큼 잊고, 
#현 시점의 정보감세 입력게이트의 중요도 만큼 곱해준 것을 더해서 
#현재 시점 기준의 memory cell값을 계산

#output gate 현재 출력값을 계산
#forget gate와 input gate에서 변경된 현재 시점의 cellstate 값을
#얼마만큼 빼내서 다음 레이어에 전달할지 결정 Ht

#데이터 로드 및 분포 확인
raw_df = pd.read_csv("/home/newuser/ML/tensorflow2/005930.KS_3MA_5MA.csv")
print(raw_df.head())
# plt.title('SAMSUNG ELECTRONIC STCOK PRICE')
# plt.ylabel('price')
# plt.xlabel('period')
# plt.grid()
# plt.plot(raw_df['Adj Close'], label='Adj Close')
# plt.show()

#데이터 전처리 - normalization - feature column label column
#비정상적으로 크거나 작은 데이터인 outlier를 적절한 값으로 바꾸거나 삭제 등 처리하는게 필요
print(raw_df.describe()) # 통계 정보 전체 확인
#거래량 volume의 최소가 0 -> 공휴일이나 주말 -> 값이 없다는 missing value로 처리
#missing value -> 누락된 데이터 - 적절한 값 조정 or 삭제
print(raw_df.isnull().sum()) #missing value 찾기

#보통 missing value는 행 전체를 삭제하는게 일반적임
print(raw_df.loc[raw_df['Open'].isna()]) 

# 먼저 0 을 NaN 으로 바꾼후, Missing Data 처리
raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)
# 각 column에 0 몇개인지 확인
for col in raw_df.columns:
    missing_rows = raw_df.loc[raw_df[col]==0].shape[0]
    print(col + ': ' + str(missing_rows))
    
# missing data 확인
raw_df = raw_df.dropna() #outlier 이나 missing value 를 다 삭제함
print(raw_df.isnull().sum())

#nomalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols =  ['Open', 'High', 'Low', 'Close', 'Adj Close',
              '3MA', '5MA', 'Volume'] # 정규화 대상 column 정의
scaled_df = scaler.fit_transform(raw_df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)
print(scaled_df)

#데이터 생성 window size 몇개의 데이터를 이용해서 정답을 만들것인지
#입력데이터인 feature와 정답 label으로 시계열 데이터 생성
#batch_size 데이터의 개수, time_steps = window size, input_dims
#fit
# 입력 파라미터 feature, label => numpy type

def make_sequene_dataset(feature, label, window_size):

    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size):

        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)


# feature_df, label_df 생성

feature_cols = [ '3MA', '5MA', 'Adj Close' ]
label_cols = [ 'Adj Close' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)



# DataFrame => Numpy 변환

feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

print(feature_np.shape, label_np.shape)

window_size = 40

X, Y = make_sequene_dataset(feature_np, label_np, window_size)

print(X.shape, Y.shape)



# train, test 분리

#split = int(len(X)*0.95)
split = -200

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# model 생성

model = Sequential()

model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()



from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=16,
          callbacks=[early_stop])



pred = model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.title('3MA + 5MA + Adj Close, window_size=40')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

plt.show()

print( np.sum(abs(y_test-pred)/y_test) / len(x_test) )




