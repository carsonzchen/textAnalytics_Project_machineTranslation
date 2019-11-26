import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, save_model
from keras import optimizers

# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

# Build prototype NMT model
def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps,units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

if __name__ == '__main__':
    data = np.loadtxt("data/data.csv", delimiter=",", dtype = 'str')
    max_length = 15

    # tokenize English text and create dictionary
    eng_tokenizer = tokenization(data[:, 0])
    eng_dict = eng_tokenizer.index_word
    with open('intermediate/eng_dict.json', 'w') as fp:
        json.dump(eng_dict, fp)
    eng_vocab = len(eng_tokenizer.word_index) + 1
    print('English Vocabulary Size: %d' % eng_vocab)

    # tokenize Spanish text and create dictionary
    esp_tokenizer = tokenization(data[:, 1])
    esp_dict = esp_tokenizer.index_word
    with open('intermediate/esp_dict.json', 'w') as fp:
        json.dump(esp_dict, fp)
    esp_vocab = len(esp_tokenizer.word_index) + 1
    print('Spanish Vocabulary Size: %d' % esp_vocab)

    # split data into train and test set
    train, test = train_test_split(data, test_size=0.05, random_state = 12)

    # prepare training data
    trainX = encode_sequences(esp_tokenizer, max_length, train[:, 1])
    trainY = encode_sequences(eng_tokenizer, max_length, train[:, 0])
    np.savetxt('intermediate/trainX.csv', trainX, delimiter=",")
    np.savetxt('intermediate/trainY.csv', trainY, delimiter=",")

    # prepare validation data
    testX = encode_sequences(esp_tokenizer, max_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, max_length, test[:, 0])
    np.savetxt('intermediate/testX.csv', testX, delimiter=",")
    np.savetxt('intermediate/testY.csv', testY, delimiter=",")

    # model compilation
    model = define_model(esp_vocab, eng_vocab, max_length, max_length, 512)
    rms = optimizers.RMSprop(lr=0.01)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')
    model.save('intermediate/model_v0.h5')