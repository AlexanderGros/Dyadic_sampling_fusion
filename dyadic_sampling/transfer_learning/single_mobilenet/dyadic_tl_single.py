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

from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D


from tensorflow.keras import models
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

from tensorflow.keras.applications import VGG16





vec_len = 4096   # 256 512 1024  2048  4096   8192  16384   32768



df = pd.read_csv('/CECI/home/users/a/g/agros/spooner/iq_bemd/cubic_simp/signal_param_all.csv', sep=',')



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
    

class DataGenerator(tf.keras.utils.Sequence):  # on   dragon2 (tested)    #DataGenerator('train', train_idx, Y_train, 32, vec_len, shuffle=True)

  def __init__(self, list_IDs, labels, batch_size, vec_len, shuffle=True, **kwargs):
    # Call to super to ensure proper initialization
    super().__init__(**kwargs)
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
    
    batch_x = (X_batch, dyadic_filter(X_batch, 2), dyadic_filter(X_batch, 4), dyadic_filter(X_batch, 8), dyadic_filter(X_batch, 16), 
               dyadic_filter(X_batch, 32) )
               
    
    batch_y = self.labels[list_IDs_temp]                
    return batch_x, batch_y
    
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
            
            
print('datagenerator definition successfull')


listing = np.arange(56000)



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


'''


# Function to increase the input dimension via a Conv2D layer
def preprocess_block(input_tensor, target_height):
    # Apply a convolution to expand the input from (2, vec_len, 1) to (target_height, vec_len, 1)
    x = Conv2D(filters=32, kernel_size=(2, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    return Reshape((target_height, vec_len, 1))(x)  # Reshape after convolution to match target_height

# Define MobileNetV2 block with pre-trained weights and custom input shape
def mobile_net_block(input_tensor, instance_name):
    base_model = MobileNetV2(input_shape=(64, vec_len, 3), include_top=False, weights='imagenet', name=instance_name)
    x = base_model(input_tensor)
    x = GlobalAveragePooling2D()(x)
    return x



# Use MobileNetV2 on the preprocessed input, give each instance a unique name
x1 = mobile_net_block(preprocessed_1, "mobilenetv2_1")
x2 = mobile_net_block(preprocessed_2, "mobilenetv2_2")
x3 = mobile_net_block(preprocessed_3, "mobilenetv2_3")
x4 = mobile_net_block(preprocessed_4, "mobilenetv2_4")
x5 = mobile_net_block(preprocessed_5, "mobilenetv2_5")
x6 = mobile_net_block(preprocessed_6, "mobilenetv2_6")



'''


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate, Conv2D, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define vec_len and other variables
vec_len = 4096
dr = 0.5  # Dropout rate example
num_classes = len(classes)

# Function to increase the input dimension via a Conv2D layer
def preprocess_block(input_tensor, target_height):
    print('np shape:', np.shape(input_tensor))
    # Apply a convolution to expand the input from (2, vec_len, 1) to (target_height, vec_len, 1)
    x = Conv2D(filters=32, kernel_size=(2, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    return Reshape((target_height, np.shape(input_tensor)[2], 1))(x)  # Reshape after convolution to match target_height

# Define MobileNetV2 block with pre-trained weights and custom input shape
def mobile_net_block(input_tensor, vec_len, instance_name):
    base_model = MobileNetV2(input_shape=(64, vec_len, 3), include_top=False, weights='imagenet', name=instance_name)  # 3 channels for RGB
    x = base_model(input_tensor)
    x = GlobalAveragePooling2D()(x)
    return x

# Define input shapes for each channel
channel_1 = Input(shape=(2, vec_len, 1), name="input1")
#channel_2 = Input(shape=(2, vec_len // 2, 1), name="input2")
#channel_3 = Input(shape=(2, vec_len // 4, 1), name="input3")
#channel_4 = Input(shape=(2, vec_len // 8, 1), name="input4")
#channel_5 = Input(shape=(2, vec_len // 16, 1), name="input5")
#channel_6 = Input(shape=(2, vec_len // 32, 1), name="input6")

# Preprocess the inputs by increasing their height to 64 before passing into MobileNetV2
preprocessed_1 = preprocess_block(channel_1, 64)
#preprocessed_2 = preprocess_block(channel_2, 64)
#preprocessed_3 = preprocess_block(channel_3, 64)
#preprocessed_4 = preprocess_block(channel_4, 64)
#preprocessed_5 = preprocess_block(channel_5, 64)
#preprocessed_6 = preprocess_block(channel_6, 64)

# Since MobileNetV2 expects 3 channels (RGB), we can replicate the single channel to 3 channels
preprocessed_1 = tf.keras.layers.Concatenate(axis=-1)([preprocessed_1] * 3)
#preprocessed_2 = tf.keras.layers.Concatenate(axis=-1)([preprocessed_2] * 3)
#preprocessed_3 = tf.keras.layers.Concatenate(axis=-1)([preprocessed_3] * 3)
#preprocessed_4 = tf.keras.layers.Concatenate(axis=-1)([preprocessed_4] * 3)
#preprocessed_5 = tf.keras.layers.Concatenate(axis=-1)([preprocessed_5] * 3)
#preprocessed_6 = tf.keras.layers.Concatenate(axis=-1)([preprocessed_6] * 3)

# Use MobileNetV2 on the preprocessed input
x1 = mobile_net_block(preprocessed_1, vec_len, "mobilenetv2_1")
x2 = mobile_net_block(preprocessed_2, vec_len // 2, "mobilenetv2_2")
x3 = mobile_net_block(preprocessed_3, vec_len // 4, "mobilenetv2_3")
x4 = mobile_net_block(preprocessed_4, vec_len // 8, "mobilenetv2_4")
x5 = mobile_net_block(preprocessed_5, vec_len // 16, "mobilenetv2_5")
x6 = mobile_net_block(preprocessed_6, vec_len // 32, "mobilenetv2_6")

# Concatenate outputs from all channels
concatenated = concatenate([x1, x2, x3, x4, x5, x6])

# Fully connected layers as in the original architecture
out = Dense(256, kernel_initializer="he_normal", activation="relu", name="dense1")(concatenated)

# Optional Dropout
# out = Dropout(dr)(out)

# Final classification layer
out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name="dense2")(out)

# Reshape the final output to match the original code structure (if needed)
out = Reshape([num_classes])(out)

# Define model with inputs from all channels
model = Model(inputs=[channel_1, channel_2, channel_3, channel_4, channel_5, channel_6], outputs=out)

# Compile the model
optim = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=["accuracy"])

# Display the model summary
model.summary()



print('end of architecture')

#%%

#!!! change here
#filepath = 'spooner_bemd_iq_cnn_model.wts.h5'
#filepath = './weights_folder/TL_mn2_df_4096'
filepath = './weights_folder/TL_mn2_df_4096.keras'


print(' ')
print('Training start')
start_time = time.time()


# perform training ...
#   - call the main training loop in keras for our network+dataset


history = model.fit(DataGenerator(train_idx, y, 32, vec_len, shuffle=True),          # DataGenerator(train_idx, y, 32, vec_len, shuffle=True),
    steps_per_epoch=len(train_idx)//batch_size, # replaces batch size
    epochs=nb_epoch,
    #show_accuracy=False,
    verbose=2,
    validation_data=DataGenerator(test_idx, y, 32, vec_len, shuffle=True),  # needs to get generator also
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pate, verbose=1, mode='auto')
    ])


print("training time:  %s seconds " % (time.time() - start_time))  # 

print('end of training')

#%%

print(' ')
print('Loading model')
# we re-load the best weights once training is finished
#model.load_weights(filepath)
model = tf.keras.models.load_model(filepath)
print('end of loading')




#%%

print(' ')
print('Evaluating model')

# Show simple version of performance
score = model.evaluate(DataGenerator(test_idx, y, 32, vec_len, shuffle=True), verbose=1)
print('end of evaluation')
print(' ')
print('score: ', score)








