import numpy as np
import pandas as pd
import pickle
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

if __name__ == '__main__':
    max_length = 10
    paramfile = 'intermediate/dim_dict.json'
    datafile = 'data/datasample.csv'
    data = np.loadtxt(datafile, delimiter=",", dtype = 'str')

    # tokenize English text and save tokenizer object
    eng_tokenizer = tokenization(data[:, 0])
    eng_vocab = len(eng_tokenizer.word_index) + 1
    print('English Vocabulary Size: %d' % eng_vocab)
    with open('intermediate/eng_tokenizer.pickle', 'wb') as handle:
        pickle.dump(eng_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # tokenize Spanish text
    esp_tokenizer = tokenization(data[:, 1])
    esp_vocab = len(esp_tokenizer.word_index) + 1
    print('Spanish Vocabulary Size: %d' % esp_vocab)

    # save dictionary dimensions as json for training
    dim_dict = {}
    dim_dict['eng_vocab'] = eng_vocab
    dim_dict['esp_vocab'] = esp_vocab
    dim_dict['max_length'] = max_length
    with open(paramfile, 'w') as fp:
        json.dump(dim_dict, fp)

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