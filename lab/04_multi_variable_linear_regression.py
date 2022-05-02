import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

model = Sequential()
#input = 1D, output = 1D
model.add(Dense(1, input_shape=(3, ), activation='linear'))
#mse = Mean Squared Error 
model.compile(optimizer=SGD(learning_rate=1e-5), loss='mse')
model.summary()

#training
hist = model.fit(x_data, y_data, epochs=1000)

y_predict = model.predict(np.array([[72., 93., 90.]]))
print(y_predict)