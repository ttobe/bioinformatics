from re import X
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam, SGD

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#reshape to tensor -> heights = 32, cross = 32, channel = 3
x_train=x_train.reshape(-1, 32, 32, 3)
x_test=x_test.reshape(-1, 32, 32, 3)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#normalization
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

cnn = Sequential()

#first conv layer filter size = 3,3, filter 32
cnn.add(Conv2D(input_shape=(32,32,3), kernel_size=(3,3),filters=32, activation='relu'))
#second conv layer filter size = 3,3, filter = 64
cnn.add(Conv2D(kernel_size=(3,3), filters=64, activation='relu'))
#third maxpooling
cnn.add(MaxPool2D(pool_size=(2,2)))
#fourth dropout
cnn.add(Dropout(0.25))
#fifth flat layer 3D tensor -> 1D vector
cnn.add(Flatten())
#sixth Denselayer (hidden state)
cnn.add(Dense(128, activation='relu'))
#seventh Dropout layer
cnn.add(Dropout(0.5))
#eighth Denselayer (output is 10)
cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss = 'sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test))

cnn.evaluate(x_test, y_test)
