from os import sep
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

xy = np.loadtxt("/home/newuser/ML/lab/data-04-zoo.csv", delimiter=",")
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

#data one-hot encoding
t_train = tf.keras.utils.to_categorical(training_t_data, num_classes=7)
t_test = tf.keras.utils.to_categorical(test_t_data, num_classes=7)

model = Sequential()
model.add(Dense(7, input_shape=(16, ), activation='softmax'))
model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(training_x_data, t_train, epochs=1000)
y_predict = model.evaluate(test_x_data, t_test)
print(y_predict)