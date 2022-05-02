from os import sep
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#use neural net->> more layer
model = Sequential()
#first layer inputD : 2, outputD : 2
model.add(Dense(2, input_shape=(2, ), activation='sigmoid'))
#second layer inputD : 2, outputD : 1
model.add(Dense(1, input_shape=(2, ), activation='sigmoid'))
model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_data, y_data, epochs=10000)

y_predict = model.predict(x_data)
print("Predict: ", y_predict)
y_accuracy = model.evaluate(x_data, y_data)
print("Cost: ", y_predict[0])
print("Accuracy: ", y_accuracy[1])