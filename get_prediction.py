import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

def get_sentence(pred_array, tokenizer):
    sentence = []
    for i in range(len(pred_array)):
        word = get_word(pred_array[i], tokenizer)
        if i > 0:
            if (word == get_word(pred_array[i-1], tokenizer)) or (word == None):
                sentence.append('')
            else:
                sentence.append(word)
        else:
            if(word == None):
                sentence.append('')
            else:
                sentence.append(word)
    return sentence

def gen_predicted_text(pred_array, tokenizer):
    alltext = []
    for pred in preds:
        s_list = get_sentence(pred, tokenizer)
        s = ' '.join(s_list).strip(' ')
        alltext.append(s)
    return alltext

filename = 'model.p1.24nov2019'
model = load_model(filename)
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))

k = gen_predicted_text(preds, eng_tokenizer)

pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : k})
pred_df.sample(20)