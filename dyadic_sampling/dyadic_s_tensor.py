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

#%%
# definitions

# Create a sample signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Function for dyadic filtering using decimation
def dyadic_filter(signal, decimation_factor):
    filtered_signal = signal[::decimation_factor]
    return filtered_signal


# -> decimator needs seperated input CNNs

#%%
#
# Perform dyadic filtering with a decimation factor of 2
filtered_signal = dyadic_filter(signal, 2)


# print and show

plt.figure()
plt.plot(signal)

plt.figure()
plt.plot(filtered_signal)

# Print the results
print("Original Signal:", signal)
print("Filtered Signal (Dyadic Filtering):", filtered_signal)




#%%
# Create a sample signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Function for dyadic filtering using moving average
def dyadic_filter(signal, scale):
    kernel = np.ones(scale) / scale
    filtered_signal = np.convolve(signal, kernel, mode='same')
    return filtered_signal

# Perform dyadic filtering with different scales
scales = [2, 4, 8]

for scale in scales:
    filtered_signal = dyadic_filter(signal, scale)
    print(np.shape(filtered_signal))
    print(f"Filtered Signal with Scale {scale}:", filtered_signal)

# -> as these have the same length we can apply 

#%%
# do we get the same output by adding them together s-like IMFs


filtered_signal_1 = dyadic_filter(signal, 2)
filtered_signal_2 = dyadic_filter(signal, 4)
filtered_signal_3 = dyadic_filter(signal, 8)

recomposed = filtered_signal_1 + filtered_signal_2 + filtered_signal_3

plt.figure()
plt.plot(signal)
plt.plot(recomposed)

# we do not get the signal back !






