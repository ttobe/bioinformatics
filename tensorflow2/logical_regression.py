from tensorboard import summary
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from datetime import datetime

import numpy as np

print(tf.__version__)

# loadtxt() 이용해서 diabetes.csv 읽어들임

import numpy as np

# loadtxt() 이용해서 diabetes.csv 읽어들임

import numpy as np

try:

    loaded_data = np.loadtxt("/home/newuser/ML/tensorflow2/diabetes.csv", delimiter=",")

    # training data / test data 분리

    seperation_rate = 0.3  # 분리 비율
    test_data_num = int(len(loaded_data) * seperation_rate)

    np.random.shuffle(loaded_data)

    test_data = loaded_data[ 0:test_data_num ]
    training_data = loaded_data[ test_data_num: ]

    # training_x_data / training_t__data 생성

    training_x_data = training_data[ :, 0:-1]
    training_t_data = training_data[ :, [-1]]

    # test_x_data / test_t__data 생성
    test_x_data = test_data[ :, 0:-1]
    test_t_data = test_data[ :, [-1]]

    print("loaded_data.shape = ", loaded_data.shape)
    print("training_x_data.shape = ", training_x_data.shape)
    print("training_t_data.shape = ", training_t_data.shape)

    print("test_x_data.shape = ", test_x_data.shape)
    print("test_t_data.shape = ", test_t_data.shape)

except Exception as err:

    print(str(err))

# Logistic Regression 을 keras 이용하여 생성
model = Sequential()

# 노드 1개인 출력층 생성
#training t data shape = 759, 1 >> shape[1] = 1!!!
#input shape[1] = 8
model.add(Dense(training_t_data.shape[1], input_shape=(training_x_data.shape[1], ), activation='sigmoid'))

# 학습을 위한 optimizer, 손실6함수 loss 정의
model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

start_time = datetime.now()
#validation 20% overfitting check verbose is output printing
hist = model.fit(training_x_data, training_t_data, epochs=500, validation_split=0.2, verbose=2)
end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)

model.evaluate(test_x_data, test_t_data)
import matplotlib.pyplot as plt

plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')

plt.legend(loc='best')

plt.show()

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend(loc='best')

plt.show()
