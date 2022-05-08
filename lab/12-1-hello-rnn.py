from os import sep
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, LSTMCell, RNN, LSTM
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist


idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]  # hihell
y_data = [[1, 0, 2, 3, 3, 4]]  # ihello

#array를 one hot data로 변환
x_one_hot = tf.keras.utils.to_categorical(x_data, num_classes=5)
print(x_one_hot)
y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=5)
print(y_one_hot)

model = Sequential()
#cell의 개수는 5개, input은 6글자니까 6, 차원은 5이니까 5
model.add(LSTM(5, input_shape=(6,5),return_sequences=True))
model.compile(optimizer=Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_one_hot, y_one_hot, epochs=50)
predictions = model.predict(x_one_hot)
print(predictions)