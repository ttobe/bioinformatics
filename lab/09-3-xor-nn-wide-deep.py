from os import sep
import os
import datetime
import tensorboard
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#use neural net->> more layer
model = Sequential()
#first layer inputD : 2, outputD : 10
model.add(Dense(2, input_shape=(2, ), activation='sigmoid'))
#2 layer outputD : 1
model.add(Dense(1, input_shape=(2, ), activation='sigmoid'))

model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#prepare callback
log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime(("%Y%m%d-%H%M%S")))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

hist = model.fit(x_data, y_data, epochs=5000, callbacks=[tensorboard_callback])

y_predict = model.predict(x_data)
print("Predict: ", y_predict)
y_accuracy = model.evaluate(x_data, y_data)
print("Cost: ", y_predict[0])
print("Accuracy: ", y_accuracy[1])