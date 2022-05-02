from os import sep
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

xy = np.loadtxt("/home/newuser/ML/lab/data-03-diabetes.csv", delimiter=",")
x_data = xy[:,0:-1]
y_data = xy[:, [-1]]

seperation_rate = 0.3
test_data_num = int(len(xy) * seperation_rate)

np.random.shuffle(xy)

test_data = xy[0:test_data_num]
training_data = xy[test_data_num : ]

# training_x_data / training_t__data 생성

training_x_data = training_data[ :, 0:-1]
training_t_data = training_data[ :, [-1]]

# test_x_data / test_t__data 생성
test_x_data = test_data[ :, 0:-1]
test_t_data = test_data[ :, [-1]]

print("loaded_data.shape = ", xy.shape)
print("training_x_data.shape = ", training_x_data.shape)
print("training_t_data.shape = ", training_t_data.shape)

print("test_x_data.shape = ", test_x_data.shape)
print("test_t_data.shape = ", test_t_data.shape)

model = Sequential()
model.add(Dense(1, input_shape=(training_x_data.shape[1], ), activation='sigmoid'))

model.compile(optimizer=SGD(learning_rate=0.01),loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(training_x_data,training_t_data,epochs=500, validation_split=0.2)

model.evaluate(test_x_data, test_t_data)