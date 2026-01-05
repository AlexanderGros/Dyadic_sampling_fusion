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

'''
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import ReLU, MaxPool2D, AvgPool2D, GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import plot_model

import tensorflow as tf
'''



#%%
# 
'''
def length_selector(data,length):
    output = data[:,:,:length]
    return output
''' 

vec_len = 4096   # 1024  2048  4096   8192  16384   32768



df = pd.read_csv('/CECI/home/users/a/g/agros/spooner/iq_bemd/cubic_simp/signal_param_all.csv', sep=',')



mods = df['modulation']
np.shape(mods)
mods = mods.tolist()
type(mods)
print(' ')


snr = df['snr']
snr = snr.tolist()                                           # between -2 and 12

bsp = df['base_symbol_period']   # bsp = base symbol period  | min = 1, max = 15  |  need to try to train for around the bsp dyadic filtered signal


#%%



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
#special for extraction of CFO parameters

# Retrieve the column
column = df['carrier_offset']
# Calculate standard deviation
std = column.std()
# Calculate maximum deviation (absolute difference from the mean)
mean = column.mean()
min_deviation = column.min()
max_deviation = column.max()

print(f"mean: {mean}")
print(f"Standard Deviation: {std}")
print(f"Minimum: {min_deviation}")
print(f"Maximum: {max_deviation}")



#%%
#hist part

#print all column names:
print(df.columns.tolist())  # ['id', 'modulation', 'base_symbol_period', 'carrier_offset', 'excess_bw', 'upsample_factor', 'downsample_factor', 'snr', 'noise_spectral_density']

plt.figure()
SNR = df['snr']
SNR.hist()
plt.savefig('SNR.png')


bsp = df['base_symbol_period']
UF = df['upsample_factor']
DF = df['downsample_factor']

#SR = 1/bsp * DF/UF                #symbol rate:  f_{sym} = (1/T_0)*(D/U).

# Apply the formula
df['symbol_rate'] = (1 / df['base_symbol_period']) * (df['downsample_factor'] / df['upsample_factor'])


plt.figure()
SR = df['symbol_rate']
SR.hist()
plt.savefig('symbol_rate.png')

#sps = Fs/Fsymb = 1/ Fsymb

df['sps'] = (1 / df['symbol_rate'])

plt.figure()
SPS = df['sps']
SPS.hist()
plt.savefig('sps.png')


# Are column 'sps' and 'base_symbol_period' the same ??

print('the same ?')
print(' ')
print( (df['base_symbol_period'] == df['sps']).all() )


"""


def indices_in_range2(index_lst, snr_lst, x, y):
    snr_selection = [index for index, value in enumerate(snr_lst) if x <= value <= y]   # returns the index (position in list) of values in lst between x and y (including x and y) in the whole snr
    # Using List Comprehension
    common_indices = [x for x in snr_selection if x in index_lst]  #checks and keeps snr if the index also exists in the index list, here test_idx
    return common_indices



index_9_12 = indices_in_range2(test_idx, snr, 9, 13)


filtered_SNR = df.loc[index_9_12, 'snr']

# Plot the histogram
plt.figure()
filtered_SNR.hist()
plt.xlabel('SNR')
plt.ylabel('Frequency')
plt.title('Histogram of SNR (Filtered)')
plt.savefig('SNR_filt.png')


plt.figure()
bsp = df['base_symbol_period']
bsp.hist()
plt.xlabel('bsp')
plt.ylabel('Frequency')
plt.title('Histogram of bsp')
plt.savefig('base_symbol_period.png')
filtered_bsp = df.loc[index_9_12, 'base_symbol_period']
plt.figure()
filtered_bsp.hist()
plt.xlabel('bsp')
plt.ylabel('Frequency')
plt.title('Histogram of base_symbol_period (Filtered)')
plt.savefig('bsp_filt.png')


plt.figure()
co = df['carrier_offset']
co.hist()
plt.xlabel('co')
plt.ylabel('Frequency')
plt.title('Histogram of co')
plt.savefig('co.png')

filtered_co = df.loc[index_9_12, 'carrier_offset']
plt.figure()
filtered_co.hist()
plt.xlabel('co')
plt.ylabel('Frequency')
plt.title('Histogram of carrier_offset (Filtered)')
plt.savefig('co_filt.png')


plt.figure()
eb = df['excess_bw']
eb.hist()
plt.xlabel('eb')
plt.ylabel('Frequency')
plt.title('Histogram of eb')
plt.savefig('eb.png')
filtered_eb = df.loc[index_9_12, 'excess_bw']
plt.figure()
filtered_eb.hist()
plt.xlabel('excess_bw')
plt.ylabel('Frequency')
plt.title('Histogram of excess_bw (Filtered)')
plt.savefig('eb_filt.png')



plt.figure()
eb = df['modulation']
eb.hist()
plt.xlabel('modulation')
plt.ylabel('Frequency')
plt.title('Histogram of modulation')
plt.savefig('modulation.png')
filtered_eb = df.loc[index_9_12, 'modulation']
plt.figure()
filtered_eb.hist()
plt.xlabel('modulation')
plt.ylabel('Frequency')
plt.title('Histogram of modulation (Filtered)')
plt.savefig('modulation_filt.png')


"""








"""

  
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

"""
