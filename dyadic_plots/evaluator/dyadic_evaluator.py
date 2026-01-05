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

import json

import time

import h5py

from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

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
# 
'''
def length_selector(data,length):
    output = data[:,:,:length]
    return output
''' 

vec_len = 4096   # 1024  2048  4096   8192  16384   32768



#df = pd.read_csv('/CECI/home/users/a/g/agros/spooner/iq_bemd/cubic_simp/signal_param_all.csv', sep=',')
df = pd.read_csv('/home/umons/eletel/agros/trsf/signal_param_all.csv', sep=',')



mods = df['modulation']
np.shape(mods)
mods = mods.tolist()
type(mods)
print(' ')


snr = df['snr']
snr = snr.tolist()                                           # between -2 and 12

bsp = df['base_symbol_period']   # bsp = base symbol period  | min = 1, max = 15  |  need to try to train for around the bsp dyadic filtered signal


'''
def indices_in_range(lst, x, y):
    return [index for index, value in enumerate(lst) if x <= value <= y]

# Example usage:
my_list = [1, 5, 8, 10, 15, 20, 25]
lower_bound = 8
upper_bound = 20

result_indices = indices_in_range(my_list, lower_bound, upper_bound)

print(f"Indices of values between {lower_bound} and {upper_bound}: {result_indices}")
'''



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
print('first y shape: ', np.shape(y))   # 112000, 8


classes = possible_mods

#print('test_idx shape', np.shape(test_idx)[0] )  # 33600

#print('y test shape', np.shape(y[test_idx]))     #33600, 8


#%%
# generator 


def normalize_per_waveform(x):
    """
    Normalize each waveform in the batch to [-1, 1] range.
    x: shape (batch_size, 2, length)
    """
    # Compute min and max across I & Q channels and time for each sample
    x_min = np.min(x, axis=(1, 2), keepdims=True)
    x_max = np.max(x, axis=(1, 2), keepdims=True)
    
    # Normalize to [0, 1], then scale to [-1, 1]
    x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
    return x_norm
    
    

def dyadic_filter(signal, decimation_factor):
    #signal has 3 dimensions
    filtered_signal = signal[:,:,::decimation_factor]
    return filtered_signal
    

class DataGenerator(tf.keras.utils.Sequence):  # on   dragon2 (tested)    #DataGenerator('train', train_idx, Y_train, 32, vec_len, shuffle=True)

  def __init__(self, list_IDs, labels, batch_size, vec_len, shuffle=True, *args, **kwargs):
    super().__init__(**kwargs)
    self.list_IDs = list_IDs
    self.labels = labels
    self.batch_size = batch_size
    self.vec_len = vec_len
    self.shuffle = shuffle
    self.on_epoch_end()
    #self.h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')
    self.h5fr = h5py.File('/home/umons/eletel/agros/trsf/spooner_full.h5','r')
    

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
    
    batch_norm = normalize_per_waveform(X_batch)
    
    batch_x = ( X_batch, dyadic_filter(X_batch, 2), dyadic_filter(X_batch, 4), dyadic_filter(X_batch, 8), dyadic_filter(X_batch, 16), 
               dyadic_filter(X_batch, 32)  )
               
    
    batch_y = self.labels[list_IDs_temp]                
    return batch_x, batch_y
    
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
            
            
print('datagenerator definition successfull')


#listing = np.arange(56000)



#%%
#/////////////////////////////////////////////////
#additional test to verify the generator for this case:

#all

# Initialize DataGenerator
data_gen = DataGenerator(test_idx, y, 32, vec_len, shuffle=True)


# Collect all IDs from one complete pass of the generator
collected_IDs = []

for i in range(len(data_gen)):
    X_batch, y_batch = data_gen[i]
    # Collect list of IDs for the current batch
    batch_indices = data_gen.indexes[i * 32: (i + 1) * 32]
    collected_IDs.extend([data_gen.list_IDs[idx] for idx in batch_indices])

# Sort collected IDs for comparison
collected_IDs = sorted(collected_IDs)

# Check if collected IDs match the original list_IDs
if collected_IDs == sorted(test_idx):
    print("Test passed: The generator iterated through all IDs exactly once.")
else:
    print("Test failed: Some IDs are missing or duplicated.")
    
    
# selection (depending on SNR)



#/////////////////////////////////////////////////


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



channel_1 = Input(shape=(2,vec_len, 1), name="input1")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_1)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_1 = Flatten()(x)

channel_2 = Input(shape=(2,vec_len//2, 1), name="input2")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_2)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_2 = Flatten()(x)

channel_3 = Input(shape=(2,vec_len//4, 1), name="input3")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_3)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_3 = Flatten()(x)

channel_4 = Input(shape=(2,vec_len//8, 1), name="input4")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_4)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_4 = Flatten()(x)

channel_5 = Input(shape=(2,vec_len//16, 1), name="input5")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_5)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_5 = Flatten()(x)

channel_6 = Input(shape=(2,vec_len//32, 1), name="input6")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_6)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_6 = Flatten()(x)



concatenated = concatenate([out_1, out_2, out_3, out_4, out_5, out_6])


out = Dense(256, kernel_initializer="he_normal", activation="relu", name="dense1")(concatenated)

#out = Dropout(dr)(out)

out = Dense( len(classes), activation='softmax', kernel_initializer='he_normal', name="dense2")(out)

out = Reshape([len(classes)])(out)



model = Model(inputs = [channel_1, channel_2, channel_3, channel_4, channel_5, channel_6], outputs = out)


optim = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=["categorical_accuracy"])


model.summary()

#print(model.summary())

#plot_model(model, "bigger_one_model.png", show_shapes=True)

print('end of architecture')

#%%

#!!! change here
#filepath = 'spooner_bemd_iq_cnn_model.wts.h5'
filepath = './weights_folder/eval_4096_norm.keras'

#filepath = '/CECI/home/users/a/g/agros/dyadic_sampling/transfer_learning/weights_folder/TL_mn2_df_4096.keras'

'''
model = tf.keras.models.load_model('filepath')

# Save in H5 format
model.save('fusion_4096.h5')
'''



print(' ')
print('Training start')
start_time = time.time()


# perform training ...
#   - call the main training loop in keras for our network+dataset


history = model.fit(DataGenerator(train_idx, y, 32, vec_len, shuffle=True),          # DataGenerator(train_idx, y, 32, vec_len, shuffle=True),
    #steps_per_epoch=len(train_idx)//batch_size, # replaces batch size   # apparently removes some errors
    epochs=nb_epoch,
    #show_accuracy=False,
    verbose=2,
    validation_data=DataGenerator(test_idx, y, 32, vec_len, shuffle=True),  # needs to get generator also
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pate, verbose=1, mode='auto')
    ])
    
'''

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.figure()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('epochs_accuracy.png', bbox_inches='tight')

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('epochs_loss.png', bbox_inches='tight')


print("training time:  %s seconds " % (time.time() - start_time))  # 

print('end of training')


#saving training data tests: 
np.save('my_history.npy',history.history)

with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
    
    
'''

#%%

print(' ')
print('Loading model')
# we re-load the best weights once training is finished
#model.load_weights(filepath)
model = tf.keras.models.load_model(filepath)

#model = tf.saved_model.load(filepath)

print('end of loading')



#model = layers.TFSMLayer('/CECI/home/users/a/g/agros/dyadic_sampling/weights_folder/fusion_4096', call_endpoint='serving_default')


#%%


print(' ')
print('Evaluating model')

# Show simple version of performance
score = model.evaluate(DataGenerator(test_idx, y, 32, vec_len, shuffle=True) , verbose=0 )   #  , verbose=1
print('end of evaluation')
print(' ')
print('score: ', score)



#commented from here till end 
"""


#%%
# tester works

def plot_confusion_matrix(cm, title='Dyadic Confusion matrix', fig='tester.png', cmap=plt.cm.Blues, labels=[], fontsize=40):
    plt.figure()
    plt.xticks(fontsize=30)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=15)
    plt.yticks(tick_marks, labels, fontsize=15)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.savefig(fig, bbox_inches='tight')
  

#%%

print(' ')
print('Prediction start')

'''
it goes trough all the test data to see in which position the 1 is -> true class
then it goes 
'''

# Plot confusion matrix
#test_Y_hat = model.predict(w_val, batch_size=batch_size)  # (size 110000 , 11 )

test_Y_hat = model.predict(DataGenerator(test_idx, y, 32, vec_len, shuffle=False), verbose=0)  # 

#test_Y_hat = test_Y_hat[:1000]

print('np.shape testy hat: ', np.shape(test_Y_hat))

print('np.shape(test_idx[:1000])[0]: ', np.shape(test_idx)[0] )



conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])  # normalized version

print('test 1')

for i in range(np.shape(test_idx)[0]):   # np.shape(test_idx)[0] = 33600
    j = list(y[test_idx[i],:]).index(1)  # true values
    #print('true: ', j)
    k = int(np.argmax(test_Y_hat[i,:]))  # predicted values
    #print('predicted: ', k)
    #print(' ')
    conf[j,k] = conf[j,k] + 1  # true-predicted
    
print('test 2')    

for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    
plot_confusion_matrix(confnorm, labels=classes)
# works till here

print('end of prediction')





def indices_in_range(lst, x, y):
    return [index for index, value in enumerate(lst) if x <= value <= y]   # returns the index (position in list) of values in lst between xand y (including x and y)

# Example usage:

my_list = [1, 5, 8, 10, 15, 20, 25]
lower_bound = 8
upper_bound = 20

result_indices = indices_in_range(my_list, lower_bound, upper_bound)
print('results index test: ', result_indices)    #[2, 3, 4, 5]   #gives back index/position !!




#%%
# Plot confusion matrix
#acc = {}
#get indices of snr ranges

# max snr = 12
# min snr = -2

'''
between -3 and 0
between 0-3
between 3-6
between 6-9
between 9-13
'''

print('test 3')

snr_range = []
for i in test_idx:
    snr_range.append(snr[i])    # get a vector of all snr values in test_idx
  
print('snr range shape: ', np.shape(snr_range) )  # all snr values of test_idx   #(33600,)
print('10 snr_range values: ', snr_range[:10])    #snr values   #[7.8834556169, 4.3907103266, 4.3907103266, 2.4434135875, 11.756309885, 11.756309885]
  
idx_snr2_0 = indices_in_range(snr_range, -2, 0)  # snr range      #indices/position
print('idx_snr2_0: ', idx_snr2_0)    #[423, 424, 425, 1331, 1332, 2213, 2214, 2853, 2854, 3768, 3769, 3770, 4027, 4028, 4029, ...]

print(' ')
print('10 first snr values index ?')
print(idx_snr2_0[:10])                    #423, 424, 425, 1331, 1332, 2213, 2214, 2853, 2854, 3768]
print(' ')
print('10 first snr values')
snr2 = np.array(snr)
print(snr2[idx_snr2_0[:10]] )             #10 first snr values [9.67501208 8.4593416  8.4593416  8.48250896 8.48250896 2.52572314 2.52572314 5.39368095 5.39368095 8.87641479]
#error : definetely not between -2 and 0

#needs to be in snr_range !

print(' ')
print('10 first snr_range values')
snr3 = np.array(snr_range)
print(snr3[idx_snr2_0[:10]] )

test_Y_hat = model.predict(DataGenerator(idx_snr2_0, y, 1, vec_len, shuffle=False), verbose=0) #batch size = 1 as number of values may change depending on selected snr range  # , verbose=1

print('test 4')






'''
# estimate classes
    
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])

print('idx_snr2_0 shape : ', np.shape(idx_snr2_0))
print('test_Y_hat shape : ', np.shape(test_Y_hat))


for i in range(np.shape(idx_snr2_0)[0]):
  j = list(y[idx_snr2_0[i],:]).index(1)   # true values
  print('j: ', j)
  k = int(np.argmax(test_Y_hat[i,:]))     # estimated values
  print('k: ', k)
  print(' ')
  conf[j,k] = conf[j,k] + 1
  
  
for i in range(0,len(classes)):
  confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
  
print('test 5')
 
plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix snr=", fig='tester2.png')
#plt.savefig('tester2snr.png', bbox_inches='tight')
    
cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
print ("Overall Accuracy: ", cor / (cor+ncor) )
#acc[snr] = 1.0*cor/(cor+ncor)
    
    
print('test 6')


#%%
# Plot confusion matrix
acc = {}
for snr in snrs:

    # extract classes @ SNR
    # test_SNRs = map(lambda x: lbl[x][1], test_idx)  # had to change
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = w_val[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]   

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
 
    plot_confusion_matrix(confnorm, labels=classes, title="Dyadic Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor+ncor) )
    acc[snr] = 1.0*cor/(cor+ncor)
    
'''

print('test 7')
    






print('end')


snr_range = []
for i in test_idx:
    snr_range.append(snr[i])
  
print('snr range shape: ', np.shape(snr_range) )  # all snr values of test_idx
  
idx_snr2_0 = indices_in_range(snr_range, 5, 10)  # snr range   # (33600,)
print('idx_snr2_0: ', idx_snr2_0)   # idx_snr2_0:  [0, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,...]

print(' ')
print('prediction for snr range')

test_Y_hat = model.predict(DataGenerator(idx_snr2_0, y, 1, vec_len, shuffle=False), verbose=0) # batch size = 1 as number of values may change depending on selected snr range






'''
1) find all index positions where snr has specific value
2) keep only the ones that exist also in the test_index
3) use these values for evaluation
'''


def indices_in_range2(index_lst, snr_lst, x, y):
    snr_selection = [index for index, value in enumerate(snr_lst) if x <= value <= y]   # returns the index (position in list) of values in lst between x and y (including x and y) in the whole snr
    print('snr_selection: ', snr_selection[:10])
    print('index_lst: ', index_lst[:10])
    # Using List Comprehension
    common_indices = [x for x in snr_selection if x in index_lst]  #checks and keeps snr if the index also exists in the index list, here test_idx
    print('common_indices: ', common_indices[:10])
    #common_indices2 = [x for x in index_lst if x in snr_lst]
    #print('common_indices2: ', common_indices2[:10])
    return common_indices
    
    



'''
between -3 and 0
between 0-3
between 3-6
between 6-9
between 9-13
'''


# lower_bound 
# upper_bound 

acc = {}
def snr_conf(test_idx, y, snr, lb, ub):

    # extract classes @ SNR
    '''
    snr_range = []
    y1 = []
    for i in test_idx:
        snr_range.append(snr[i])
        y1.append(y[i])
    '''   
    #idx_snr = indices_in_range(snr_range, lb, ub)  #issues with index !!!
    idx_snr = indices_in_range2(test_idx, snr, lb, ub)  
    #test_X_i = w_val[np.where(np.array(test_SNRs)==snr)]
    #y1 = np.array(y1)
    #print('y1: ', y1[:10])
    #test_Y_i = y[idx_snr]  #orig y
    #print('test_Y_i: ', test_Y_i[:10])
    
    selected_vectors = y[idx_snr]
    label_counts = np.sum(selected_vectors, axis=0)
    print('label_counts: ', label_counts)
    
    print('idx_snr: ', idx_snr[:10])
    print('y: ', y[:10])

    # estimate classes
    test_Y_hat = model.predict(DataGenerator(idx_snr, y, 1, vec_len, shuffle=False) , verbose=0 )    #  , verbose=1
    
    score = model.evaluate(DataGenerator(idx_snr, y, 1, vec_len, shuffle=True) , verbose=0 )   #  , verbose=1
    print('score: ', score)
    
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    
    for i in range(np.shape(idx_snr)[0]):
        j = list(y[idx_snr[i],:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
 
    #plot_confusion_matrix(confnorm, labels=classes, title="Dyadic Confusion Matrix (SNR=%d)"%(snr))
    plot_confusion_matrix(confnorm, labels=classes, title="Dyadic Confusion Matrix Function (SNR=%d-%d)"%(lb, ub), fig="Dyadic_Confusion_Matrix_Function_(SNR=%d-%d).png"%(lb, ub) )  
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    oa = cor / (cor+ncor)
    print ("Overall Accuracy: ", oa )
    
    print("Confusion Matrix:\n", conf)
    print("Correctly Classified Instances (cor):", cor)
    print("Total Instances (np.sum(conf)):", np.sum(conf))
    print("Misclassified Instances (ncor):", ncor)
    
    print(' ')
    
    return oa

    

#snr_conf(test_idx, y, snr, 9, 12)




print(' ')


oa_1 = snr_conf(test_idx, y, snr, -3, 0)
oa_2 = snr_conf(test_idx, y, snr, 0, 3)
oa_3 = snr_conf(test_idx, y, snr, 3, 6)
oa_4 = snr_conf(test_idx, y, snr, 6, 9)
oa_5 = snr_conf(test_idx, y, snr, 9, 13)

    
  
#%%
# !!!
# Plot accuracy curve

snr_hist = [-1.5, 1.5, 4.5, 7.5, 10.5]
acc_hist = [oa_1, oa_2, oa_3, oa_4, oa_5]

print('acc_hist: ', acc_hist )

# orig 20 fontsize and 6 for linewidth
plt.figure()
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30)
#plt.plot(snr, list(map(lambda x: acc[x], snr_hist)), linewidth=9 )
plt.plot(snr_hist, acc_hist, linewidth=9 )
plt.xlabel("Signal to Noise Ratio", fontsize=30)
plt.ylabel("Classification Accuracy", fontsize=30)
plt.title("Dyadic Classification Accuracy ", fontsize=30)
plt.savefig('snr_curve.png', bbox_inches='tight')


print(' ')
print(' ')
print('end of code')



oa_1 = snr_conf(test_idx, y, snr, -3, 0)
oa_2 = snr_conf(test_idx, y, snr, 0, 3)
oa_3 = snr_conf(test_idx, y, snr, 3, 6)
oa_4 = snr_conf(test_idx, y, snr, 6, 13)
#oa_5 = snr_conf(test_idx, y, snr, 9, 13)

    
  
#%%
# !!!
# Plot accuracy curve

snr_hist = [-1.5, 1.5, 4.5, 9]
acc_hist = [oa_1, oa_2, oa_3, oa_4]

print('acc_hist: ', acc_hist )

# orig 20 fontsize and 6 for linewidth
plt.figure()
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30)
#plt.plot(snr, list(map(lambda x: acc[x], snr_hist)), linewidth=9 )
plt.plot(snr_hist, acc_hist, linewidth=9 )
plt.xlabel("Signal to Noise Ratio", fontsize=30)
plt.ylabel("Classification Accuracy", fontsize=30)
plt.title("Dyadic Classification Accuracy ", fontsize=30)
plt.savefig('snr_curve2.png', bbox_inches='tight')


print(' ')
print(' ')
print('end of code')


"""
