from os import sep
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

model = Sequential()
model.add(Dense(3, input_shape=(3, ), activation='softmax'))
model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

hist = model.fit(x_data, y_data, epochs=1000)
print("Prediction: ", model.predict_classes(x_test))
print("Accuracy: ", model.evaluate(x_test,y_test)[1])