import numpy as np
import matplotlib.pyplot as plt
import json
import os

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, save_model
from keras import optimizers

# Build prototype NMT model
def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units, dropout=0.2))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True, dropout=0.2))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

# Model Evaluation
def loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train','validation'])
    plt.savefig('Loss history.png')
    plt.close()

# This script is designed to run in GPU environment
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Use specific GPU
    
    modelfile = 'models/model_v0.h5'
    trainX = np.loadtxt("intermediate/trainX.csv", delimiter=",")
    trainY = np.loadtxt("intermediate/trainY.csv", delimiter=",")
    
    paramfile = 'intermediate/dim_dict.json'
    with open(paramfile, 'r') as f:
        dim_dict = json.load(f)
    
    # model compilation
    model = define_model(dim_dict['esp_vocab'], dim_dict['eng_vocab'], dim_dict['max_length'], dim_dict['max_length'], 512)
    ado = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=ado, loss='sparse_categorical_crossentropy')

    # define checkpint
    checkpoint = ModelCheckpoint(modelfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # train model
    history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                        epochs=50, batch_size=256, validation_split=0.2, callbacks=[checkpoint], 
                        verbose=1)
    model.save(modelfile)

    # plot loss
    loss_plot(history)