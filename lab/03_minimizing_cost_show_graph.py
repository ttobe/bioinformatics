import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

x_train = [1,2,3]
y_train = [1,2,3]

model = Sequential()
#input = 1D, output = 1D
model.add(Dense(1, input_shape=(1, ), activation='linear'))
#mse = Mean Squared Error 
model.compile(optimizer=SGD(learning_rate=0.1), loss='mse')
model.summary()

#training
hist = model.fit(x_train, y_train, epochs=200)

plt.plot(hist.history['loss'], label='train loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()