from os import sep
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist

#data 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 학습 데이터 / 테스트 데이터 정규화 (Normalization) = (data - Min) / (Max - Min) range: 0 ~ 1
x_train = (x_train - 0.0) / (255.0 - 0.0)
x_test = (x_test - 0.0) / (255.0 - 0.0)
# 정답 데이터 원핫 인코딩 (One-Hot Encoding) output 0~9 -> 10
# in tf1, one_hot= True in read_data_sets("MNIST_data/", one_hot=True)
# took care of it, but here we need to manually convert them
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


print(x_train.shape) # 60000, 28 , 28 >> 28*28 input
print(y_train.shape) # 60000, 10

model = Sequential()
#1: flatten input
model.add(Flatten(input_shape=(28,28)))
#2: hidden layer units = 256
model.add(Dense(256, activation='relu'))
#3: output layer units = 10
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train, epochs=100, batch_size=100)
predictions = model.predict(x_test)
print('Prediction: \n', predictions)

score = model.evaluate(x_train, y_train)
print('Accuracy: ', score[1])

#94%