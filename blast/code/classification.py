import numpy as np
sdnum=1122
np.random.seed(sdnum)

import random as rn
rn.seed(sdnum)

import tensorflow as tf
tf.random.set_seed(sdnum)

import h5py
import scipy.io
import re

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Conv2D

from utils import formatDeepMirTar2, padding, convert3D, flatten

from sklearn.model_selection import train_test_split

# prepare the input data
# hsa-mir-6509,UUUUUGUGUGUGAAAUUAGGUAGUGGCAGUGGAACACUAUAUUAAUCAGGUUUCCACUGCCACUACCUAAUUUCUCAGAUGGAAA,FALSE
# miRNA 이름, pre-miRNA, label
data_path = "data/classification.csv"

seqs = list()
l = list()
# encode = dict(zip('NAUGC', range(5)))
with open(data_path, 'r') as infl:
	next(infl)
	lines = infl.read().split('\n') # 구분자를 엔터로 나눈거 한 라인들!
	for line in lines[:10000]:
		miRNA_ID, miRNA, label = line.split(',') # 구분자 ,로 나눠줌
		miRNA = re.sub('T', 'U',miRNA)
		label = label.rstrip('\r\n')
		# token = [encode[s] for s in miRNA.upper()]
		seqs.append(miRNA)
		if label == 'FALSE':
			l.append(0)
		else:
			l.append(1)


# x = [x[0] for x in seqs]
x = seqs
print(x[0])
y = [int(y) for y in l]

input_token_index = {'A': 0, 'U': 1, 'G': 2, 'C': 3} # input 문자 인덱스
x_2 = np.zeros((len(x), 164, 4), dtype='float32') # 19000 * 29 * 4 0으로 채워진 배열 선언
print('x_2.shape',x_2.shape)
for i in range(1123): # 1124개의 데이터중에
    for j in range(len(x[i])):
        # 그냥 원핫 인코딩
        x_2[i, j, input_token_index[x[i][j]]] = 1. 
print(x_2[0])
y_2 = np.array(y).reshape(len(y), 1)


percT = 0.2
X_train, X_test, y_train, y_test = train_test_split(x_2, y_2, test_size=percT, random_state=sdnum)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=sdnum)
input_token_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3, } # input 문자 인덱스

print("X TRAIN: ",X_train.shape)
print(X_train[0])
print("X valid: ",X_valid.shape)
print("X test: ",X_test.shape)
print("Y_train: ", y_train.shape)
print(y_train[0])
epochs = 30
batch = 6
learning_rate = 0.001
dropout = 0.5
fils = 128
ksize = 12

acc = 0

model = Sequential()
model.add(Dense(4,input_shape=(164,4,1), dtype="float32"))
model.add(Conv2D(filters=fils, kernel_size=(6,4), activation='relu'))
model.add(Conv2D(filters=fils, kernel_size=(6,4), activation='relu'))
model.add(Conv2D(filters=fils, kernel_size=(6,4), activation='relu'))
model.add(LSTM(100, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
adam = optimizers.Adam(learning_rate)
model.summary()
model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_valid, y_valid), verbose=2)
			

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))