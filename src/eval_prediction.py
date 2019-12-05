import pandas as pd
import numpy as np
import pickle
import os
import re

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

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

def gen_predicted_text(preds_array, tokenizer):
    alltext = []
    for pred in preds_array:
        s_list = get_sentence(pred, tokenizer)
        s = re.sub("\s\s+", " ", ' '.join(s_list))
        alltext.append(s)
    return alltext

def gen_sent_bleu(orig_s, pred_s):
    orig_list = [token for token in word_tokenize(orig_s)]
    pred_list = [token for token in word_tokenize(pred_s)]
    score = sentence_bleu([orig_list], pred_list, weights=(1, 0, 0, 0))
    return score

if __name__ == "__main__":
    # Load model
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Use specific GPU
    modelfile = '../models/model_v0.h5'
    model = load_model(modelfile)

    # Load test data sets
    testX = np.loadtxt("../intermediate/testX.csv", delimiter=",")
    testY = np.loadtxt("../intermediate/testY.csv", delimiter=",")
    #testX = testX[0:101] # Save computational resources by looking at only first 100
    #testY = testY[0:101] # Save computational resources by looking at only first 100

    # Load english dictionary for looking up
    engdict_file = '../intermediate/eng_tokenizer.pickle'
    with open(engdict_file, 'rb') as handle:
        eng_tokenizer = pickle.load(handle)

    # Predict results and convert to original text
    preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
    pred_text = gen_predicted_text(preds, eng_tokenizer)
    orig_text = gen_predicted_text(testY, eng_tokenizer)

    # Print a sample of predicted text
    pred_df = pd.DataFrame({'actual' : orig_text, 'predicted' : pred_text})
    pred_df['bleu'] = pred_df.apply(lambda x: gen_sent_bleu(x.actual, x.predicted), axis=1)
    pred_df.to_csv('../results/sample_output.csv', index= False)