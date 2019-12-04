import string
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_text(filename):
    """ Read text line by line into list of sentences """
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.readlines()
    file.close()
    return text

def gen_parallel_data(text_output, text_input):
    """ Construct dataset in the form of paired list for each parallel sentence """
    combined = list(map(list, zip(text_output, text_input)))
    return combined

def preprocess_nopd(data):
    """ Execute selected steps of text preprocessing """
    # Remove punctuation, digits, and make lower case
    digits = string.digits
    punctuations = string.punctuation + '\n'
    removables = digits + punctuations
    data[:,0] = [s.translate(str.maketrans('', '', removables)).lower() for s in data[:,0]]
    data[:,1] = [s.translate(str.maketrans('', '', removables)).lower() for s in data[:,1]]
    return data

def count_words_string(s):
    """ Count number of words of a sting, based on whitespaces"""
    return len(re.findall(r'\w+', s))

def restrict_sent_length(data, maxlength = 20, restrict_input = 1):
    """ Output only rows of dataset containing text with number of words fewer than maxlength"""
    data_short = []
    if restrict_input == 1:
        col = 1
    else:
        col = 0
    for line in data:
        if count_words_string(line[col]) <= maxlength:
            data_short.append(line)
    return data_short

def save_prepro_chunks(data, filepath, n = 100000):
    """ Process parallel text data by chunk of n and write as an array on disk for modeling """
    if os.path.exists(filepath):
        os.remove(filepath)
    for i in range(0, len(data), n):
        data_chunk = np.array(data[i:i + n])
        processed_data = preprocess_nopd(data_chunk)
        if os.path.exists(filepath):
            with open(filepath, 'at', encoding='utf-8') as f:
                np.savetxt(f, processed_data, delimiter=",", fmt='%s')
        else:
            with open(filepath, 'wt', encoding='utf-8') as f:
                np.savetxt(f, processed_data, delimiter=",", fmt='%s')

if __name__ == '__main__':
    text_en = read_text("data/europarl-v7.es-en.en")
    text_es = read_text("data/europarl-v7.es-en.es")
    datafull = gen_parallel_data(text_en, text_es)
    datasample = restrict_sent_length(datafull, maxlength = 10)
    datasample5 = datasample*5
    #save_prepro_chunks(datafull, 'data/data.csv')
    save_prepro_chunks(datasample, 'data/datasample.csv')