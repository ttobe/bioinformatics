import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np

print(tf.__version__)

# train data 
# t_data = 2*x1 -3*x2 + 2*x3

x_data = np.array([ [1, 2, 0], [5, 4, 3], [1, 2, -1], [3, 1, 0], [2, 4, 2], 
                    [4, 1, 2], [-1, 3, 2], [4, 3, 3], [0, 2, 6], [2, 2, 1],
                    [1, -2, -2], [0, 1, 3], [1, 1, 3], [0, 1, 4], [2, 3, 3] ])

t_data = np.array([-4, 4, -6, 3, -4, 
                   9, -7, 5, 6, 0,
                   4, 3, 5, 5, 1])

print('x_data.shape = ', x_data.shape, ', t_data.shape = ', t_data.shape)

# [2] 모델 (model) 구축
model = Sequential()
model.add(Dense(1,input_shape=(3, ), activation='linear'))

# [3] 모델 (model) 컴파일 및 summary
model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse') 
model.summary()
hist = model.fit(x_data, t_data, epochs=1000)
