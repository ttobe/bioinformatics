from tensorboard import summary
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from datetime import datetime

import numpy as np

print(tf.__version__)

x_data = np.array([2, 4, 6, 8, 10,
                   12, 14, 16, 18, 20]).astype('float32')

y_data = np.array([0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1]).astype('float32')

model = Sequential()
#total two dense
#hidden state node = 8
model.add(Dense(8, input_shape=(1, ), activation='sigmoid'))
#ouput state node = 1
model.add(Dense(1, activation='sigmoid'))
#H
model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#training
hist = model.fit(x_data, y_data, epochs = 500)

test_data = np.array([0.5, 3.0, 3.5, 
                      11.0, 13.0, 31.0])

sigmoid_value = model.predict(test_data)

logical_value = tf.cast(sigmoid_value > 0.5,
                      dtype=tf.float32)

for i in range(len(test_data)):
    print(test_data[i], 
          sigmoid_value[i], 
          logical_value.numpy()[i])

model.weights