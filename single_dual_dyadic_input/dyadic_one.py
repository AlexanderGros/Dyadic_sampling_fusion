# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:55:38 2024

@author: Alexander

dyadic filtering using sampling method
"""

#%%
# imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd

import time

import h5py

import keras
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *

from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import ReLU, MaxPool2D, AvgPool2D, GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# import keras_tuner as kt


from tensorflow.keras.models import load_model

from tensorflow.keras.utils import plot_model

import tensorflow as tf



#%%
# definitions



#%%
# 
'''
def length_selector(data,length):
    output = data[:,:,:length]
    return output
''' 

vec_len = 4096   # 256 1024  2048  4096   8192  16384   32768



df = pd.read_csv('/CECI/trsf/umons/eletel/agros/signal_param_all.csv', sep=',')



mods = df['modulation']
np.shape(mods)
mods = mods.tolist()
type(mods)
print(' ')


snr = df['snr']
snr = snr.tolist()



np.random.seed(2023)
#n_examples = X.shape[0]
n_examples = 112000  #full spooner data size
n_train = int(n_examples * 0.7)  # Why a 50/50 split instead of 60/40 or 70/30??  yeah good question  !!! for spooner change to 70-30 !

train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False) 
# -> random selection for training (a shuffle is performed by keras !)
test_idx = list(set(range(0,n_examples))-set(train_idx))



def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1]) 
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

possible_mods = ['bpsk', 'qpsk', 'dqpsk', '8psk', 'msk', '16qam', '64qam', '256qam']    # there are 8 modulations in the spooner dataset

#Y_train = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), train_idx))) # only get the modulation

#Y_test = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), test_idx)))

y = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), np.arange(n_examples))))


#print('first y_train shape: ', np.shape(Y_train))
print('first y shape: ', np.shape(y))


classes = possible_mods





#%%
# generator 

def dyadic_filter(signal, decimation_factor):
    #signal has 3 dimensions
    filtered_signal = signal[:,:,::decimation_factor]
    return filtered_signal
    
    

class DataGenerator(keras.utils.Sequence):  # on   dragon2 (tested)    #DataGenerator('train', train_idx, Y_train, 32, vec_len, shuffle=True)

  def __init__(self, list_IDs, labels, batch_size, vec_len, shuffle=True):
    self.list_IDs = list_IDs
    self.labels = labels
    self.batch_size = batch_size
    self.vec_len = vec_len
    self.shuffle = shuffle
    self.on_epoch_end()
    self.h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')

  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))
	

  def __getitem__(self, index):
  
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    list_IDs_temp = sorted(list_IDs_temp)
    
    
    # Retrieve data
    X_batch = self.h5fr['spooner'][list_IDs_temp,:,:vec_len]
    
    '''
    batch_x = [X_batch, dyadic_filter(X_batch, 2), dyadic_filter(X_batch, 4), dyadic_filter(X_batch, 8), dyadic_filter(X_batch, 16), 
               dyadic_filter(X_batch, 32)]
    '''
    #batch_x = dyadic_filter(X_batch, 4)
    
    #batch_x = [X_batch, dyadic_filter(X_batch, 2), dyadic_filter(X_batch, 4)]
    
    batch_x = X_batch
    
    batch_y = self.labels[list_IDs_temp]                
    return batch_x, batch_y
    
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
            
            
print('datagenerator definition successfull')




#%%

# Set up some model params 
nb_epoch = 100   # number of epochs to train on (orig 100)
batch_size = 32  # training batch size
pate = 10



#%%
# architecture
# called bigger

print('start AI architecture definition')

dr=0.5

tf.keras.backend.clear_session()


channel_1 = Input(shape=(2, vec_len, 1), name="input1")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_1)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_1 = Flatten()(x)

'''

channel_2 = Input(shape=(2, vec_len//2, 1), name="input2")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_2)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_2 = Flatten()(x)



channel_3 = Input(shape=(2, vec_len//4, 1), name="input3")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_3)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_3 = Flatten()(x)




channel_4 = Input(shape=(2, vec_len//8, 1), name="input4")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_4)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_4 = Flatten()(x)



channel_5 = Input(shape=(2, vec_len//16, 1), name="input5")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_5)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_5 = Flatten()(x)



channel_6 = Input(shape=(2, vec_len//32, 1), name="input6")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_6)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_6 = Flatten()(x)
'''



#concatenated = concatenate([out_1, out_2, out_3, out_4, out_5, out_6])

#concatenated = concatenate([out_1, out_2, out_3])


#out = Dense(128, kernel_initializer="he_normal", activation="relu", name="dense1")(concatenated) #adapt here !!!

out = Dense(128, kernel_initializer="he_normal", activation="relu", name="dense1")(out_1) 

#out = Dense(22, kernel_initializer="he_normal", activation="relu", name="dense1")(out_1) 

#out = Dropout(dr)(out)

out = Dense( len(classes), activation='softmax', kernel_initializer='he_normal', name="dense2")(out)

out = Reshape([len(classes)])(out)



#model = Model(inputs = [channel_1, channel_2, channel_3, channel_4, channel_5, channel_6], outputs = out)  #adapt here !!!

model = Model(inputs = [channel_1, channel_2, channel_3], outputs = out) 


optim = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=["accuracy"])


model.summary()

#print(model.summary())

#plot_model(model, "bigger_one_model.png", show_shapes=True)

print('end of architecture')

#%%

#!!! change here
#filepath = 'spooner_bemd_iq_cnn_model.wts.h5'
filepath = './weights_folder/fusion_32678_orig_only'



print(' ')
print('Training start')
start_time = time.time()


# perform training ...
#   - call the main training loop in keras for our network+dataset


history = model.fit(DataGenerator(train_idx, y, 32, vec_len, shuffle=True),  
    steps_per_epoch=len(train_idx)//batch_size, # replaces batch size
    epochs=nb_epoch,
    #show_accuracy=False,
    verbose=2,
    validation_data=DataGenerator(test_idx, y, 32, vec_len, shuffle=True),  # needs to get generator also
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=pate, verbose=1, mode='auto')
    ])


print("training time:  %s seconds " % (time.time() - start_time))  # 

print('end of training')



#%%

print(' ')
print('Loading model')
# we re-load the best weights once training is finished
#model.load_weights(filepath)
model = keras.models.load_model(filepath)
print('end of loading')




#%%

print(' ')
print('Evaluating model')

# Show simple version of performance
score = model.evaluate(DataGenerator(test_idx, y, 32, vec_len, shuffle=True), verbose=1)
print('end of evaluation')
print(' ')
print('score: ', score)








