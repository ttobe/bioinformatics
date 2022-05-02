import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

xy = np.loadtxt("/home/newuser/ML/lab/data-01-test-score.csv", delimiter=",")
x_data = xy[:,0:-1]
y_data = xy[:, [-1]]


print(xy, "xy = ", xy.shape)
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

model = Sequential()
model.add(Dense(1, input_shape=(3, ), activation='linear'))
model.compile(optimizer=SGD(learning_rate=1e-5), loss='mse')
model.summary()

hist = model.fit(x_data, y_data, epochs=5000)

y_predict = model.predict(np.array([[100, 70, 101]]))
print(y_predict)