import pandas as pd
import numpy as np
import pickle
import json
import string

from keras.models import load_model
from src.eval_prediction import get_word, get_sentence, gen_predicted_text
from src.prepare_nn import encode_sequences

def read_text_input(s):
    slist = s.splitlines()
    return slist

def reduce_nopd(data):
    """ Execute selected steps of text preprocessing """
    # Remove punctuation, digits, and make lower case
    digits = string.digits
    punctuations = string.punctuation + '\n'
    removables = digits + punctuations
    newdata = [s.translate(str.maketrans('', '', removables)).lower() for s in data]
    return np.array(newdata)

def conv_sent_vector(tokenizer, length, input_string):
    ti = read_text_input(input_string)
    feed_array = reduce_nopd(ti)
    vector = encode_sequences(tokenizer, length, feed_array)
    return vector

def translate_sentences(modelfile, input_string):
    model = load_model(modelfile)
    paramfile = 'intermediate/dim_dict.json'
    # Load dictionary dimensions json file
    with open(paramfile, 'r') as f:
        dim_dict = json.load(f)
    length = dim_dict['max_length']

    # Load spanish dictionary for converting sentences
    espdict_file = 'intermediate/esp_tokenizer.pickle'
    with open(espdict_file, 'rb') as handle1:
        esp_tokenizer = pickle.load(handle1)

    # Load english dictionary for looking up
    engdict_file = 'intermediate/eng_tokenizer.pickle'
    with open(engdict_file, 'rb') as handle2:
        eng_tokenizer = pickle.load(handle2)

    sent_vector = conv_sent_vector(esp_tokenizer, length, input_string)
    preds = model.predict_classes(sent_vector.reshape((sent_vector.shape[0],sent_vector.shape[1])))
    pred_text = gen_predicted_text(preds, eng_tokenizer)
    return pred_text