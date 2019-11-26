import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers

#filename = 'model.p1.25nov2019'
modelfile = 'model_v0.h5'
model = load_model(modelfile)
trainX = np.loadtxt("intermediate/trainX.csv", delimiter=",")
trainY = np.loadtxt("intermediate/trainY.csv", delimiter=",")

checkpoint = ModelCheckpoint(modelfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# train model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=20, batch_size=512, validation_split = 0.2, callbacks=[checkpoint], 
                    verbose=1)

# ### Model Evaluation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()
plt.close()