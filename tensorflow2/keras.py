import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

import numpy as np

print(tf.__version__)

# [1] 데이터셋 생성

x_data = np.array([1, 2, 3, 4, 5, 6]) #input data
t_data = np.array([3, 4, 5, 6, 7, 8]) #test data

# [2] 모델 (model) 구축
model = Sequential()
model.add(Flatten(input_shape=(1,)))
model.add(Dense(1, activation='linear'))

# [3] 모델 (model) 컴파일 및 summary
model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse') 
model.summary()

hist = model.fit(x_data, t_data, epochs=1000)

result = model.predict(np.array([-3.1, 3.0, 3.5, 15.0, 20.1]))
print(result)