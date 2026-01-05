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
#from matplotlib import rc

#import mplcyberpunk

import scipy.signal
import pandas as pd

import time

import h5py

#import keras  #for keras.sequence.utils

import tensorflow as tf

#rc('text', usetex=True)
#rc('font', family='serif')
#rc('font', size=10.0)
#rc('legend', fontsize=20.0)
#rc('font', weight='bold')
# plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'





#%%
# 


vec_len = 1024   # 1024  2048  4096   8192  16384   32768



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

Y_train = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), train_idx))) # only get the modulation

#Y_test = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), test_idx)))

y = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), np.arange(n_examples))))


print('first y_train shape: ', np.shape(Y_train))
print('first y shape: ', np.shape(y))


classes = possible_mods





#%%
# generator 

def dyadic_filter(signal, decimation_factor):
    #signal has 3 dimensions
    filtered_signal = signal[:,:,::decimation_factor]
    return filtered_signal
    

class DataGenerator(tf.keras.utils.Sequence):  # on   dragon2 (tested)    #DataGenerator('train', train_idx, Y_train, 32, vec_len, shuffle=True)

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
    
    batch_x = [X_batch, dyadic_filter(X_batch, 2), dyadic_filter(X_batch, 4), dyadic_filter(X_batch, 8), dyadic_filter(X_batch, 16), 
               dyadic_filter(X_batch, 32)]
               
    
    batch_y = self.labels[list_IDs_temp]                
    return batch_x, batch_y
    
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
            
            
print('datagenerator definition successfull')


#%%


def dyadic_filter(signal, decimation_factor):
    #signal has 3 dimensions
    filtered_signal = signal[:,:,::decimation_factor]
    return filtered_signal







z = DataGenerator(train_idx, y, 32, vec_len, shuffle=True)

x, y = next(iter(z))


print('len x', len(x) )               #  6
print('shape x[0]', np.shape(x[0]) )  # (32, 2, 8192)




#%%
#no cyberpunk but final plot

fig = plt.figure()
gs = fig.add_gridspec(6, hspace=0)
axs = gs.subplots(sharex=False, sharey=True)
fig.suptitle('Dyadic Down-sampled IQ Signal', fontsize=20)
axs[0].plot(x[0][0,0,:], linewidth=2.0)
axs[1].plot(x[1][0,0,:], linewidth=2.0)
axs[2].plot(x[2][0,0,:], linewidth=2.0)
axs[3].plot(x[3][0,0,:], linewidth=2.0)
axs[4].plot(x[4][0,0,:], linewidth=2.0)
axs[5].plot(x[5][0,0,:], linewidth=2.0)
axs[-1].set_xlabel('Signal samples', fontsize=20)  # -1 to plot below last one
axs[3].set_ylabel('Amplitude', fontsize=20)

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer() 
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=10, width=5, length = 10)
plt.tight_layout()



plt.savefig('clean_final.png', bbox_inches='tight')



"""



#%%


plt.style.use("cyberpunk") #add cyberpunk effect


plt.figure()
plt.plot(x[0][0,0,:], linewidth=3.0, label='signal}')
plt.tight_layout()
plt.savefig('test.png', bbox_inches='tight')

plt.figure()
plt.plot(x[1][0,0,:], linewidth=3.0, label='signal}')
plt.tight_layout()
mplcyberpunk.add_glow_effects()
plt.savefig('test2.png', bbox_inches='tight')


plt.figure()
plt.plot(x[0][0,0,:], linewidth=3.0, label='signal}')
plt.plot(x[1][0,0,:], linewidth=3.0, label='dyadic by 2')
plt.plot(x[0][2,0,:], linewidth=3.0, label='dyadic by 4')
plt.plot(x[0][3,0,:], linewidth=3.0, label='dyadic by 8')
plt.title('Dyadic sampling scheme', fontsize=30)
plt.xlabel('Signal sample', fontsize=30)
plt.ylabel('Amplitude', fontsize=30)
plt.tick_params(axis='x', labelsize=20, width=7, length = 10)
plt.tick_params(axis='y', labelsize=20, width=7, length = 10)
plt.legend()
plt.tight_layout()


mplcyberpunk.add_glow_effects(gradient_fill=True)

plt.savefig('test4.png', bbox_inches='tight')



fig = plt.figure()
gs = fig.add_gridspec(6, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Hilbert IMFs real part', fontsize=30)
axs[0].plot(x[0][0,0,:], linewidth=2.0)
axs[1].plot(x[1][0,0,:], linewidth=3.0)
axs[1].plot(x[2][0,0,:], linewidth=3.0)
axs[2].plot(x[3][0,0,:], linewidth=5.0, color="C1")
axs[3].plot(x[4][0,0,:], linewidth=5.0)
axs[4].plot(x[5][0,0,:], linewidth=5.0)
axs[-1].set_xlabel('Signal sample', fontsize=30)  # -1 to plot below last one
axs[3].set_ylabel('Amplitude', fontsize=30)

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()
    ax.tick_params(axis='x', labelsize=20, width=7, length = 10)
    ax.tick_params(axis='y', labelsize=20, width=7, length = 10)
plt.tight_layout()

mplcyberpunk.add_glow_effects(gradient_fill=True)

plt.savefig('test99.png', bbox_inches='tight')



plt.style.use("cyberpunk")
fig = plt.figure()
gs = fig.add_gridspec(6, hspace=0)
axs = gs.subplots(sharex=False, sharey=True)
fig.suptitle('Dyadic downsampled IQ signal', fontsize=20)
axs[0].plot(x[0][0,0,:], linewidth=2.0)
axs[1].plot(x[1][0,0,:], linewidth=2.0, color="C1")
axs[2].plot(x[2][0,0,:], linewidth=2.0)
axs[3].plot(x[3][0,0,:], linewidth=2.0, color="C1")
axs[4].plot(x[4][0,0,:], linewidth=2.0)
axs[5].plot(x[5][0,0,:], linewidth=2.0, color="C1")
axs[-1].set_xlabel('Signal samples', fontsize=20)  # -1 to plot below last one
axs[3].set_ylabel('Amplitude', fontsize=20)

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()
    #ax.tick_params(axis='x', labelsize=10, width=5, length = 10)  
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=10, width=5, length = 10)
plt.tight_layout()

#mplcyberpunk.add_glow_effects(axs[0], gradient_fill=True)
#mplcyberpunk.add_gradient_fill(axs[0], alpha_gradientglow=0.7)
mplcyberpunk.add_gradient_fill(axs[0], alpha_gradientglow=0.7)
mplcyberpunk.add_gradient_fill(axs[1], alpha_gradientglow=0.7)
mplcyberpunk.add_gradient_fill(axs[2], alpha_gradientglow=0.7)
mplcyberpunk.add_gradient_fill(axs[3], alpha_gradientglow=0.7)
mplcyberpunk.add_gradient_fill(axs[4], alpha_gradientglow=0.7)
mplcyberpunk.add_gradient_fill(axs[5], alpha_gradientglow=0.7)

plt.savefig('test999.png', bbox_inches='tight')

plt.savefig('test999.svg', bbox_inches='tight')



plt.figure()
plt.plot(x[0][1,0,:], linewidth=3.0, label='signal}')
plt.tight_layout()
mplcyberpunk.add_glow_effects()
plt.savefig('0.png', bbox_inches='tight')

plt.figure()
plt.plot(x[1][1,0,:], linewidth=3.0, label='signal}')
plt.plot(x[2][1,0,:], linewidth=3.0, label='signal}')
plt.tight_layout()
mplcyberpunk.add_glow_effects()
plt.savefig('1.png', bbox_inches='tight')

plt.figure()
plt.plot(x[3][1,0,:], linewidth=3.0, label='signal}')
plt.plot(x[4][1,0,:], linewidth=3.0, label='signal}')
plt.tight_layout()
mplcyberpunk.add_glow_effects()
plt.savefig('2.png', bbox_inches='tight')




plt.style.use("cyberpunk")
zzz = np.arange(0, np.pi*2, 0.05) 
  
# Using built-in trigonometric function we can directly plot 
# the given cosine wave for the given angles 
Y1 = np.sin(zzz) 
Y2 = np.cos(zzz) 
Y3 = np.tan(zzz) 
Y4 = np.tanh(zzz) 
  
# Initialise the subplot function using number of rows and columns 
figure, axis = plt.subplots(2, 2) 
mplcyberpunk.add_glow_effects()
  
# For Sine Function 
axis[0, 0].plot(Y1) 
axis[0, 0].set_title("Sine Function") 


# For Cosine Function 
axis[0, 1].plot(Y2, color="C1") 
axis[0, 1].set_title("Cosine Function") 

  
# For Tangent Function 
axis[1, 0].plot(Y3) 
axis[1, 0].set_title("Tangent Function") 

  
# For Tanh Function 
axis[1, 1].plot(Y4, color="C1") 
axis[1, 1].set_title("Tanh Function") 


mplcyberpunk.add_gradient_fill(axis[0, 0], alpha_gradientglow=0.3)
mplcyberpunk.add_gradient_fill(axis[0, 1], alpha_gradientglow=0.5)
mplcyberpunk.add_gradient_fill(axis[1, 0], alpha_gradientglow=0.7)
mplcyberpunk.add_gradient_fill(axis[1, 1], alpha_gradientglow=0.9)
plt.savefig('zzzz.png', bbox_inches='tight')






"""





