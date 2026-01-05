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
from tensorflow.keras.layers import Reshape, concatenate, Conv2D
from tensorflow import cast, float32
import tensorflow as tf


input_x = 2
input_y = 50

# input sizes
def create_matrix(x, y):
    return np.arange(1, x * y + 1).reshape(x, y)

input_tensor = create_matrix(input_x, input_y)

print('shape:', np.shape(input_tensor))
print('input_tensor: ', input_tensor[:,:10])

input_tensor = input_tensor.reshape(1, input_x, input_y, 1)
print('shape input_tensor :', np.shape(input_tensor))

input_tensor = cast(input_tensor, dtype=float32)  #tensorflow (conv2D) works with float32 and when creating data it is int64 

# Function to increase the input dimension via a Conv2D layer
def preprocess_block_1(input_tensor, target_height, kernels):  #filters = number of channels
    # Apply a convolution to expand the input from (2, vec_len, 1) to (target_height, vec_len, 1)
    x = Conv2D(filters=32, kernel_size=(2, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    print('fct shape x:', np.shape(x))
    print(x)
    x2 = Reshape((target_height, np.shape(input_tensor)[2], 1))(x)  # Reshape after convolution to match target_height
    print('fct shape x2:', np.shape(x2))
    return x2
    
    
    
    
yo = preprocess_block_1(input_tensor, 64, 25)

print('end of test')














import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate
from tensorflow import cast, float32

input_x = 2
input_y = 5

# Create input matrix
def create_matrix(x, y):
    return np.arange(1, x * y + 1).reshape(x, y)

input_tensor = create_matrix(input_x, input_y)

print('Original Matrix Shape:', np.shape(input_tensor))
print('Original Matrix (first 10 elements):\n', input_tensor[:, :10])

# Reshape the input tensor
input_tensor = input_tensor.reshape(1, input_x, input_y, 1)
print('Reshaped Input Tensor Shape:', np.shape(input_tensor))

# Cast to float32 for TensorFlow compatibility
input_tensor = cast(input_tensor, dtype=float32)

# Function to increase the input dimension via a Conv2D layer
def preprocess_block_1(input_tensor, target_height, kernels):
    # Apply a convolution to expand the input
    x = Conv2D(filters=kernels, kernel_size=(2, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    print('Shape after Conv2D:', np.shape(x))
    
    # Print the output of Conv2D to see the transformation
    tf.print('Output of Conv2D:', x)

    # Concatenate the original input tensor with the new kernel output along the channel axis
    x_combined = Concatenate(axis=-1)([input_tensor, x])  # Concatenating the original input tensor with the Conv2D output
    print('Shape after Concatenation:', np.shape(x_combined))
    tf.print('Output of Concatenation:', x_combined[0,0,:,0])
    tf.print('Output of Concatenation:', x_combined[0,1,:,0])
    tf.print('Output of Concatenation:', x_combined[0,0,:,15])
    
    # Reshape after concatenation if necessary (example: if you want to adjust the output shape)
    x2 = Reshape((target_height, np.shape(x_combined)[2], 1))(x_combined)
    print('Shape after Reshape:', np.shape(x2))
    tf.print('Output of x2:', x2[0,0,:,0])
    tf.print('Output of x2:', x2[0,1,:,0])
    tf.print('Output of x2:', x2[0,2,:,0])
    tf.print('Output of x2:', x2[0,3,:,0])
    tf.print('Output of x2:', x2[0,4,:,0])
    tf.print('Output of x2:', x2[0,63,:,0])
    
    print('where 1: ', tf.where(x2 == 1) )
    print('where 2: ', tf.where(x2 == 2) )
    print('where 3: ', tf.where(x2 == 3) )
    print('where 4: ', tf.where(x2 == 4) )
    print('where 5: ', tf.where(x2 == 5) )
    print('where 6: ', tf.where(x2 == 6) )
    print('where 7: ', tf.where(x2 == 7) )
    print('where 8: ', tf.where(x2 == 8) )
    print('where 9: ', tf.where(x2 == 9) )
    print('where 10: ', tf.where(x2 ==10) )

    
    
    return x2

# Call the preprocess function with 31 kernels and keep the original data as the first channel
yo = preprocess_block_1(input_tensor, 64, 31)  # 31 kernels to create additional channels

print('End of test')





values_to_find = tf.constant(np.arange(1, 11))

# To store the positions of found values
positions = []

# Iterate over each value and find its positions
for value in values_to_find:
    # Use tf.where to find the positions of the current value in the tensor
    indices = tf.where(tensor == value)
    
    # If the value exists in the tensor, append the indices to positions list
    if tf.size(indices) > 0:
        positions.append((value.numpy(), indices.numpy()))

# Print the positions of each value from 1 to 10
for value, pos in positions:
    print(f"Value {value} found at positions (indices): {pos}")
















