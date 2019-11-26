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

def preprocess_nopunc(data):
    """ Execute selected steps of text preprocessing """
    # Remove punctuation and make lower case
    punctuations = string.punctuation + '\n'
    data[:,0] = [s.translate(str.maketrans('', '', punctuations)).lower() for s in data[:,0]]
    data[:,1] = [s.translate(str.maketrans('', '', punctuations)).lower() for s in data[:,1]]
    return data

def save_prepro_chunks(data, filepath, n = 100000):
    """ Process parallel text data by chunk of n and write as an array on disk for modeling """
    if os.path.exists(filepath):
        os.remove(filepath)
    for i in range(0, len(data), n):
        data_chunk = np.array(data[i:i + n])
        processed_data = preprocess_nopunc(data_chunk)
        if os.path.exists(filepath):
            with open(filepath, 'at', encoding='utf-8') as f:
                np.savetxt(f, processed_data, delimiter=",", fmt='%s')
        else:
            with open(filepath, 'wt', encoding='utf-8') as f:
                np.savetxt(f, processed_data, delimiter=",", fmt='%s')

def show_sentence_len_dist(data):
    """ Show frequency distribution of sentence length for the dataset """
    output_l = []
    input_l = []

    for i in data[:,0]:
        output_l.append(len(i.split()))
    for i in data[:,1]:
        input_l.append(len(i.split()))
    length_df = pd.DataFrame({'output_lang':output_l, 'input_lang':input_l})
    length_df.hist(bins = 30)
    plt.show()
    plt.close()

if __name__ == '__main__':
    text_en = read_text("data/europarl-v7.es-en.en")
    text_es = read_text("data/europarl-v7.es-en.es")
    datafull = gen_parallel_data(text_en, text_es)
    save_prepro_chunks(datafull, 'data/data.csv')