import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

x_train = [1,2,3,4,5]
y_train = [2.1,3.1,4.1,5.1,6.1]

model = Sequential()
#input = 1D, output = 1D
model.add(Dense(1, input_shape=(1, ), activation='linear'))
#mse = Mean Squared Error 
model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
model.summary()

#training
hist = model.fit(x_train, y_train, epochs=200)

t_predict = model.predict(np.array([5,2.5,1.5,3.5]))
print(t_predict)

print(model.input)
print(model.output)
print(model.weights)