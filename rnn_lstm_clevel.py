from __future__ import print_function

import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed


from RNN_utils import *


DATA_DIR = './text.txt'
BATCH_SIZE = 50
HIDDEN_DIM = 500
SEQ_LENGTH = 25
WEIGHTS = ''
MODE = 'train'
GENERATE_LENGTH = 300



#Function to generate text
def generate_text

#Function to load the data
def load_data(data_dir, seq_length):
	data = open(data_dir, 'r').read()
	chars = list(set(data))
	VOCAB_SIZE = len(chars)

	print('Data length: {} characters'.format(len(data)))
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

	X = np.zeros((len(data)/seq_length, seq_length, VOCAB_SIZE))
	y = np.zeros((len(data)/seq_length, seq_length, VOCAB_SIZE))
	
	for i in range(0, len(data)/seq_length):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char

# Creating training data
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


nb_epoch = 0

while True:
	print('\n\nEpoch: {}\n'.format(nb_epoch))
	model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
	nb_epoch += 1

	#CALL FUNCTION TO GENERATE TEXT
	generate_text

		if nb_epoch % 10 == 0:
			model.save_weights('hidden_dimension_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))
