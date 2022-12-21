import numpy as np
import random as rn
import tensorflow as tf

import h5py
import scipy.io

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Concatenate, Dot, Conv1D, \
    MaxPooling1D, Dropout, Activation, Permute, Attention
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
import tensorflow.keras as K

from utils import formatDeepMirTar2, padding, convert3D, flatten

from sklearn.model_selection import train_test_split


class BahdanauAttention(K.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.W1 = Dense(self.units)
        self.W2 = Dense(self.units)
        self.V = Dense(1)

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config

    def call(self, inputs, training=None):  # 단, key와 value는 같음
        values = inputs[0]
        query = inputs[1]
        # query shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class ReductionLayer(K.layers.Layer):
    def __init__(self, **kwargs):
        super(ReductionLayer, self).__init__()

    def call(self, inputs, training=None):
        return tf.reduce_sum(inputs, axis=1)


class ExpandLayer(K.layers.Layer):
    def __init__(self, **kwargs):
        super(ExpandLayer, self).__init__()

    def call(self, inputs, training=None):
        return tf.expand_dims(inputs, 1)


sdnum = 1122
np.random.seed(sdnum)
rn.seed(sdnum)
tf.random.set_seed(sdnum)

# 바꾼 내용 적용 여부 flag (코드 변경시 여기서 변경사항을 제어할 수 있도록 하기)
USE_ONE_HOT_VECTOR = F
miRNA_REVERSE_PADDING = True
mRNA_REVERSE_SEQ = True
TEST_ONLY_BEST_HYPER_PARAMETER_IN_PAPER = True
DOT_PRODUCT_OR_BAHDANAU = "bahdanau"

# prepare the input data
inputf = "data/data_DeepMirTar_miRAW_noRepeats_3folds.txt"
seqs, label = formatDeepMirTar2(inputf, reverse_padding=miRNA_REVERSE_PADDING, reverse_mrna=mRNA_REVERSE_SEQ)

x = [x[0] for x in seqs]
x = padding(x)  # pad 0 at end of input
y = [int(y) for y in label]

x_2 = x.reshape(x.shape[0], x.shape[1])
y_2 = np.array(y).reshape(len(y), 1)
print(x_2[1])
print(y_2.shape)
percT = 0.2
X_train, X_test, y_train, y_test = train_test_split(x_2, y_2, test_size=percT, random_state=sdnum)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=percT, random_state=sdnum)

# use one-hot vector instead of embedding
if USE_ONE_HOT_VECTOR:
    X_train = np.eye(5)[X_train]
    X_valid = np.eye(5)[X_valid]
    X_test = np.eye(5)[X_test]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(X_train.shape[1])

# best original(in-paper) hyper-parameter = [learning-rate = 0.005, dropout = 0.2, batch-size = 100]
epochs = 1000
if TEST_ONLY_BEST_HYPER_PARAMETER_IN_PAPER:
    batches = [100]
    learning_rate = [0.005]
    dropout = [0.2]
else:
    # batches = [10, 30, 50, 100, 200]
    batches = [50, 100, 200]
    # learning_rate = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    learning_rate = [0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
fils = 320
ksize = 12

acc = 0
for batch in batches:
    for lr in learning_rate:
        for dout in dropout:
            # used Functional API instead of Sequential API to implement attention
            seq_input = Input(shape=(79,), dtype="int32")
            # remove embedding layer
            if not USE_ONE_HOT_VECTOR:
                embed = Embedding(input_dim=5, output_dim=5, input_length=X_train.shape[1])(seq_input)
                conv = Conv1D(filters=fils, kernel_size=ksize, activation='relu')(embed)
            else:
                conv = Conv1D(filters=fils, kernel_size=ksize, activation='relu')(seq_input)
            conv_d = Dropout(dout)(conv)
            maxp = MaxPooling1D(pool_size=2)(conv_d)
            maxp_d = Dropout(dout)(maxp)
            bi_lstm, fh, fc, bh, bc = Bidirectional(
                LSTM(32, dropout=dout, activation='relu', return_sequences=True, return_state=True))(maxp_d)
            hidden_state = Concatenate()([fh, bh])

            if DOT_PRODUCT_OR_BAHDANAU == "dot":
                # hidden_with_time_axis = Lambda(lambda x: tf.expand_dims(x, 1))(hidden_state)
                hidden_with_time_axis = ExpandLayer()(hidden_state)
                # score = Lambda(lambda x: hidden_with_time_axis * x)(bi_lstm)
                score = Dot(axes=2)([hidden_with_time_axis, bi_lstm])
                # attention_weights = K.activations.softmax(score, axis=1)
                attention_weights = Activation('softmax')(score)
                # context_vector = Lambda(lambda x: attention_weights * x)(bi_lstm)
                bi_lstm_p = Permute((2, 1))(bi_lstm)
                context_vector = Dot(axes=2)([attention_weights, bi_lstm_p])
                # context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
                context_vector = ReductionLayer()(context_vector)
                dense = Dense(16, activation='relu')(context_vector)
            elif DOT_PRODUCT_OR_BAHDANAU == "bahdanau":
                attention = BahdanauAttention(32)
                context_vector, attention_weights = attention([bi_lstm, hidden_state])
                dense = Dense(16, activation='relu')(context_vector)
            elif DOT_PRODUCT_OR_BAHDANAU == "no":
                dense = Dense(16, activation='relu')(hidden_state)
            else:
                raise (Exception("Select 'dot' or 'bahdanau' only."))

            dense_d = Dropout(dout)(dense)
            output = Dense(1, activation='sigmoid')(dense_d)

            model = Model(inputs=seq_input, outputs=output)
            model.summary()
            adam = Adam(lr)
            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

            es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.001, verbose=1, patience=100)
            mcp = ModelCheckpoint(
                filepath='results/custom/rev_pad_seq_b_attention/miTAR_CNN_BiRNN_b' + str(batch) + '_lr' + str(
                    lr) + '_dout' + str(
                    dout) + '.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_valid, y_valid), verbose=2,
                      callbacks=[es, mcp])

            bestModel = load_model(
                'results/custom/rev_pad_seq_b_attention/miTAR_CNN_BiRNN_b' + str(batch) + '_lr' + str(
                    lr) + '_dout' + str(
                    dout) + '.h5', custom_objects={'ReductionLayer': ReductionLayer, 'ExpandLayer': ExpandLayer,
                                                   'BahdanauAttention': BahdanauAttention})
            paras = [batch, lr, dout]
            print(f"Fitting done. Evaluating...")
            scores = bestModel.evaluate(X_test, y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))

            # 결과 출력 방식 개선
            if scores[1] > acc:
                acc = scores[1]
                print("best so far, acc=", acc, " paras=", paras)

            print("finish paras at: batch=", batch, " lr=", lr, " dout=", dout)
# 학습 끝나고 최선의 hyper parameter와, 그때의 accuracy 출력
print(f'learning finish, best parameter = {paras}, acc = {acc}')
