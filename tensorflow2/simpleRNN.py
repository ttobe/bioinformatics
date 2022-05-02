#RNN y = 0.5sini(x)-cos(x/2)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# x = 0 ~ 100, unit-> 0.1 >>>> 1000ro
x = np.arange(0, 100, 0.1)
y = 0.5*np.sin(2*x) - np.cos(x/2.0)
#for 3D tensor input, reshape(-1, 1) >> (1000, 1) array
seq_data = y.reshape(-1, 1)
print(seq_data.shape)
print(seq_data[:5])

def seq2dataset(seq, window, horizon):
    X = [] #list for saving input data 
    Y = [] #list for saving correct data


w = 20 #window size
h = 1 #horizon factor next one predict
X,Y = seq2dataset(seq_data,w,h)