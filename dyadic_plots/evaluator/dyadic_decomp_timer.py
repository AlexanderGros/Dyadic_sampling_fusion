# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:55:38 2024

@author: Alexander

dyadic filtering using sampling method - decomposition or scaler timer
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





#%%
# 
'''
def length_selector(data,length):
    output = data[:,:,:length]
    return output
''' 

vec_len = 32768   # 1024  2048  4096   8192  16384   32768
ne = 32000


h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')

X = h5fr['spooner'][:ne+1,:,:vec_len]

print('X shape: ', np.shape(X))

#%%
# generator 

def dyadic_filter(signal, decimation_factor):
    #signal has 3 dimensions, in this test 2
    #filtered_signal = signal[:,:,::decimation_factor]
    filtered_signal = signal[:,::decimation_factor]
    return filtered_signal
    

#/////////////////////////////////////////////////




# %%
# preparing datasets

# ne =  number of elements

def data_prepper_dyadic(data, ne):
    
    #w = np.zeros((ne,2,32768,5)) # changed to 7 here ! and removed np.uint8      20
    #print('w shape zeros: ', np.shape(w))
       
    for i in range(ne):  # use smaller values for testing purposes # orig np.shape(X_train)[0]
      for j in range(1, 6):  # Loop from 1 to 5
          df = 2 ** j
          #print( 'data ne shape: ', np.shape(data[ne]) )
          temp = dyadic_filter(data[ne], decimation_factor=df)
          #print( 'temp shape: ', np.shape(temp) )
          #print( 'temp ', temp[0,:12] )
          #print('ne: ', ne)
          
      #w[ne,:,:,j] = temp

    #return  w









start_time = time.time()
z = data_prepper_dyadic(X, ne)  # 40000
print("dyadic decomposition time:  %s seconds " % (time.time() - start_time))
print('z shape :', np.shape(z))

#print( 'w_dyadic shape: ', np.shape(w_dyadic) )


print('orig :', X[ne,0,:64] )  # ok works for 1

#check with increased ne


print('end of code')










