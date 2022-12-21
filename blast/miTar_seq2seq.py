# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers, regularizers
import os
from tensorflow.keras.utils import plot_model
import numpy as np
import re

batch_size = 50  # Batch size for training.
epochs = 1000  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space. 512개의 단어
num_samples = 19000  # Number of samples to train on.
# Path to the data txt file on disk. 데이터당
data_path = '../data/mitar2.csv'
# Vectorize the data. 벡터화를 하자
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n') # 구분자를 엔터로 나눈거 한 라인들!
for line in lines[:7001]:
    # 샘플의 갯수보다 설정 샘플 갯수보다 크지 않게 min을 때린 듯
    miRNA_ID,target_text,mRNA_Accession_Number,input_text,label = line.split('\t') # 구분자 ,로 나눠줌
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character. 
    # 논문에서 했던대로 target은 \t로 시작하고, \n으로 끝나게 해준다는거
    input_text = re.sub('T', 'U',input_text)
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text) # 뒤에거는 인풋
    target_texts.append(target_text) # 앞에거는 \t ~ \n 인 타겟
    for char in input_text: # 없던 문자가 나오면 추가하기
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text: # 없던 문자가 나오면 추가하기
        if char not in target_characters:
            target_characters.add(char)
            
input_characters = sorted(list(input_characters)) # 정렬한 문자
target_characters = sorted(list(target_characters)) # 정렬한 문자
num_encoder_tokens = len(input_characters) # 문자의 개수
num_decoder_tokens = len(target_characters) # 문자의 개수
max_encoder_seq_length = max([len(txt) for txt in input_texts]) # input중에 가장 긴 길이
max_decoder_seq_length = max([len(txt) for txt in target_texts]) # target중에 가장 긴 길이
print('Number of samples:', len(input_texts)) # 샘플의 개수는 19000 설정값이당
print('Number of unique input tokens:', num_encoder_tokens) # input 문자의 개수 : 4
print('Number of unique output tokens:', num_decoder_tokens) # output 문자의 개수 : 6 
print('Max sequence length for inputs:', max_encoder_seq_length) # 제일 긴 input : 29
print('Max sequence length for outputs:', max_decoder_seq_length) # 제일 긴 output : 28
input_token_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3} # input 문자 인덱스
target_token_index = {'\t': 4, '\n': 5, 'A': 0, 'C': 1, 'G': 2, 'U': 3} # target 문자 인덱스 탭과 엔터가 들어가 있음
encoder_input_data = np.zeros( # 19000 * 29 * 4 0으로 채워진 배열 선언
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros( # 19000 * 28 * 6 0으로 채워진 배열 선언
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros( # 19000 * 28 * 6 0으로 채워진 배열 선언
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)): # 인덱스와 원소로 이루어진 터플 반환
    for t, char in enumerate(input_text):
        # 그냥 원핫 인코딩
        encoder_input_data[i, t, input_token_index[char]] = 1. 
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.      
            
# Define an input sequence and process it. 
# 그림 왼쪽에 있는 인코더 모델 작성
encoder_inputs = Input(shape=(None, num_encoder_tokens)) # n, n, 4
cnn = Conv1D(128,8, activation='relu') # n, n, 128
cnn_output =cnn(encoder_inputs)
#dropout_layer = Dropout(0.5)
#decoder_outputs = dropout_layer(decoder_outputs)
encoder_dense_1 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001))
encoder_dense_output = encoder_dense_1(cnn_output)
# LSTM
encoder = LSTM(latent_dim, return_state=True,recurrent_dropout=0.4, dropout = 0.1)
encoder_outputs, state_h, state_c = encoder(encoder_dense_output)
# We discard `encoder_outputs` and only keep the states. 
# 인코더의 스테이트만 가지고 갈거에용
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
# 오른쪽 그림 디코더 모델 작성
decoder_inputs = Input(shape=(None, num_decoder_tokens))
#cnn_decoder = Conv1D(128,8, activation='relu')
#cnn_decode_output =cnn_decoder(decoder_inputs)
#dropout_layer = Dropout(0.5)
#decoder_outputs = dropout_layer(decoder_outputs)
decoder_dense_1 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001))
decoder_dense_output = decoder_dense_1(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
# 초기값은 인코더의 lstm state값으로 하자
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,recurrent_dropout=0.4, dropout = 0.1)
decoder_outputs, _, _ = decoder_lstm(decoder_dense_output,
                                     initial_state=encoder_states)
#dropout_layer = Dropout(0.5)
#decoder_outputs = dropout_layer(decoder_outputs)
#decoder_dense_1 = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001))
#decoder_outputs = decoder_dense_1(decoder_outputs)
decoder_dense = Dense(num_decoder_tokens, activation='softmax',kernel_regularizer=regularizers.l2(0.001))
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
print(model.summary())           
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['mae', 'acc'])
callbacks = [keras.callbacks.TensorBoard(log_dir='output/my_log_dir')]

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2, callbacks=callbacks)
          
model.save_weights('../results/s2s_mitar_70.h5')

# model.load_weights('../data/s2s_mitar_100.h5')

# new weights
# model.load_weights('../data/s2s_batch_50_mitar.h5')
# scores = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# Next: inference mode (sampling).
# sampling 모델
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# input을 인코드하고 초기값을 검색
# 2) run one step of decoder with this initial state
# 초기값으로 디코더 1스텝 하기
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states
# Define sampling models
cnn_output = cnn(encoder_inputs) # 인코더 n, n, 4 인풋으로 conv 1D 생성하기
encoder_dense_output = encoder_dense_1(cnn_output) # 128 Dense layer생성
encoder_outputs, state_h, state_c = encoder(encoder_dense_output) # LSTM layer 생성해서
encoder_states = [state_h, state_c] # 인코더의 스테이트만 가지고
encoder_model = Model(encoder_inputs, encoder_states) # 모델 생성하기


#cnn_decode_output =cnn_decoder(decoder_inputs)
#decoder_dense_output = decoder_dense_1(cnn_decode_output)
decoder_dense_output = decoder_dense_1(decoder_inputs) # n,n,6 input에 128 Dense layer
decoder_state_input_h = Input(shape=(latent_dim,)) # LSTM 인풋의 n, 512
decoder_state_input_c = Input(shape=(latent_dim,)) # LSTM 인풋의 n, 512
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# 디코더 lstm 만들기
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_dense_output, initial_state=decoder_states_inputs) 
# 디코더 스테이트 저장
decoder_states = [state_h, state_c]
#decoder_outputs = dropout_layer(decoder_outputs)
#decoder_outputs = decoder_dense_1(decoder_outputs)
decoder_outputs = decoder_dense(decoder_outputs) # 6 Dense layer
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
    
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    # 특정 시퀀스를 인코드해서 스테이트 값을 뽑아낸다.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    # 1, 1, 6 짜리 배열 만들어낸다.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    # 디코드의 타겟 0, 0, 4가 이 항상 \t으로 시작한다.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # 배치 사이즈가 1이라고 가정한다.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # state도 h와 c에 저장한다.
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        # 끝내기 조건 마지막 까지 갔거나 \n이 나오면 멈추기
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [h, c]
    return decoded_sentence
canonical_site = 0
for seq_index in range(10000):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    # print('-')
    # print('Input sentence:', input_texts[seq_index])
    # print('Target sentence:', target_texts[seq_index][1:-1])
    # print('Decoded sentence:', decoded_sentence[:-1])
    # print('decoded seed region:', decoded_sentence[-8:-2], ', target seed region:',target_texts[seq_index][-8:-2])
    # # fetch decoded
    with open('../results/decoded.fasta','a+') as aa:
        aa.write(str(decoded_sentence[:-1])+'\n')
        
    #compare seed region to Canonical site types
    # if target_texts[seq_index][-8:-2] in decoded_sentence[-8:-2]:
    #     canonical_site = canonical_site + 1

print("canonical_site match: ", canonical_site)
