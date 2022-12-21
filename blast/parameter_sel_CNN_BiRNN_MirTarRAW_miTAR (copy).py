import numpy as np
sdnum=1122
np.random.seed(sdnum)

import random as rn
rn.seed(sdnum)

import tensorflow as tf
tf.random.set_seed(sdnum)

import h5py
import scipy.io

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from utils import formatDeepMirTar2, padding, convert3D, flatten

from sklearn.model_selection import train_test_split


# prepare the input data
data_path = 'data/clean_data.csv'

seqs = list()
l = list()
wrong_miRNA = []
index = 0
encode = dict(zip('NAUCG', range(5)))
with open(data_path, 'r') as infl:
	next(infl)
	lines = infl.read().split('\n') # 구분자를 엔터로 나눈거 한 라인들!
	for line in lines[:10000]:
		miRNA, mRNA = line.split(',') # 구분자 ,로 나눠줌
		mRNA = mRNA.rstrip('\r\n')
		wrong_miRNA.append(miRNA)
		if len(miRNA) < 28:
			miRNA = miRNA + 'N' * (28 - len(miRNA))
		if len(mRNA) < 29:
			mRNA = mRNA + 'N' * (29 - len(mRNA))
		seq2 = miRNA + mRNA
		token = [encode[s] for s in seq2.upper()]
			#print('seq=' + str(seq2))
		seqs.append((token,1))
		l.append(1)
	for line in lines[10000:19000]:
		index = index + 1
		miRNA, mRNA = line.split(',') # 구분자 ,로 나눠줌
		mRNA = mRNA.rstrip('\r\n')
		miRNA = wrong_miRNA[index]
		if len(miRNA) < 28:
			miRNA = miRNA + 'N' * (28 - len(miRNA))
		if len(mRNA) < 29:
			mRNA = mRNA + 'N' * (29 - len(mRNA))
		seq2 = miRNA + mRNA
		token = [encode[s] for s in seq2.upper()]
			#print('seq=' + str(seq2))
		seqs.append((token,1))
		l.append(0)

x = [x[0] for x in seqs]
x = padding(x)
y = [int(y) for y in l]



x_2 = np.array(x).reshape(x.shape[0], x.shape[1])
y_2 = np.array(y).reshape(len(y), 1)

percT = 0.2
X_train, X_test, y_train, y_test = train_test_split(x_2, y_2, test_size=percT, random_state=sdnum)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=percT, random_state=sdnum)
print("X TRAIN: ",X_train.shape)
print(X_train[0])
print("X valid: ",X_valid.shape)
print("X test: ",X_test.shape)
print("Y_train: ", y_train.shape)
print(y_train)
epochs = 100
batches = [100]
learning_rate = [0.005]
dropout = [0.3, 0.2, 0.4] 
fils = 320
ksize = 12

acc = 0
save_batch= []
save_lr = []
save_dout = []
save_acc = []
for batch in batches:
	for lr in learning_rate:
		for dout in dropout:
			model = Sequential()
			model.add(Embedding(input_dim=5, output_dim=5, input_length=x.shape[1]))
			model.add(Conv1D(filters=fils, kernel_size=ksize, activation='relu'))
			model.add(Dropout(dout))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Dropout(dout))
			model.add(Bidirectional(LSTM(32, activation='relu')))
			model.add(Dropout(dout))
			model.add(Dense(16, activation='relu'))
			model.add(Dropout(dout))
			model.add(Dense(1, activation='sigmoid'))

			model.summary()
			adam = optimizers.Adam(lr)
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

			# es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.001, verbose=1, patience=100)
			# mcp = ModelCheckpoint(filepath='/home/newuser/ML/miTAR/mitar/results/miTAR_CNN_BiRNN_b' + str(batch) + '_lr' + str(lr) + '_dout' + str(dout) + '.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)
			lstm = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_valid, y_valid), verbose=2)
			hist = model.history

			# bestModel = load_model('/home/newuser/ML/miTAR/mitar/results/miTAR_CNN_BiRNN_b' + str(batch) + '_lr' + str(lr) + '_dout' + str(dout) + '.h5')

			scores = model.evaluate(X_test, y_test, verbose=0)
			print("Accuracy: %.2f%%" % (scores[1]*100))

			save_batch.append(batch)
			save_lr.append(lr)
			save_dout.append(dout)
			save_acc.append(scores[1]*100)

			if scores[1] > acc:
				acc = scores[1]
				paras = [batch, lr, dout]
				print("best so far, acc=", acc, " paras=", paras)


print("finish paras at: batch=", batch, " lr=", lr, " dout=", dout)

for i in range(len(save_acc)):
	print("batch=",save_batch[i],"lr=",save_lr[i],"dout=",save_dout[i],"acc=",save_acc[i])

