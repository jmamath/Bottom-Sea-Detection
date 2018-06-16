#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:44:17 2018

@author: root
"""

#################### 0 - IMPORT RELEVANT LIBRARIES ####################
import scipy.io as sio
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D 
from keras import initializers
from keras import regularizers 
from sklearn.model_selection import train_test_split


from utils_1 import *

#################### 1 - EXPERIMENT 1 #################### 

########## 1.1 - Loading the data sets 

## loading data
data = sio.loadmat('Conv_data.mat', squeeze_me=True)

## 2011 data
X_train_1 = data['X_train_2011']
Y_train_1 = data['Y_train_2011']


## 2015 data
X_2015 = data['X_train_2015']
Y_2015 = data['Y_train_2015']

## Spliting dev and test sets
X_dev_1, X_test_1, Y_dev_1, Y_test_1 = train_test_split(X_2015, Y_2015 ,test_size=0.5,shuffle = False)

#################### 1.2 - Convolutional Neural Network #################### 
# define base model
def conv_model():
    # create model
    model = Sequential()
    model.add(Conv1D(input_shape=[120,4], filters=1, kernel_size=20, strides=1, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=20, strides=1, padding='valid'))  
    model.add(BatchNormalization(axis=-1))
    model.add(Flatten())
    model.add(Dense(4, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

cnn = conv_model()
cnn.summary()

history = cnn.fit(X_train_1, Y_train_1, epochs = 26, batch_size = 1024, shuffle=True)

## Debugging
losses = history.history["loss"]

epoch = history.epoch
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.scatter(epoch, losses)
plt.show()

##### Computing the loss
training_error_1 = cnn.evaluate(X_train_1,Y_train_1)
dev_error_1 = cnn.evaluate(X_dev_1,Y_dev_1)

##### Generating predicted cleanbottom
Y_pred_train_1_cnn = cnn.predict(X_train_1)
Y_pred_dev_1_cnn = cnn.predict(X_dev_1)

#################### 2 - EXPERIMENT 2 #################### 

########## 2.1 - Loading the data sets 
## 2011 data
X_train_2 = data['X_train_2015']
Y_train_2 = data['Y_train_2015']


## 2015 data
X_2011 = data['X_train_2011']
Y_2011 = data['Y_train_2011']

## Spliting dev and test sets
X_dev_2, X_test_2, Y_dev_2, Y_test_2 = train_test_split(X_2011, Y_2011 ,test_size=0.5,shuffle = False)

########## 2.2 - Convolutional Neural Network #################### 
# define base model
def conv_model():
    # create model
    model = Sequential()
    model.add(Conv1D(input_shape=[120,4], filters=1, kernel_size=20, strides=1, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=20, strides=1, padding='valid'))  
    model.add(BatchNormalization(axis=-1))
    model.add(Flatten())
    model.add(Dense(4, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

cnn = conv_model()
cnn.summary()

history2 = cnn.fit(X_train_2, Y_train_2, epochs = 26, batch_size = 1024, shuffle=True)

## Debugging
losses = history2.history["loss"]

epoch = history2.epoch
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.scatter(epoch, losses)
plt.show()

##### Computing the loss
training_error_2 = cnn.evaluate(X_train_2,Y_train_2)
dev_error_2 = cnn.evaluate(X_test_2,Y_dev_2)

##### Generating predicted cleanbottom
Y_pred_train_2_cnn = cnn.predict(X_train_2)
Y_pred_dev_2_cnn = cnn.predict(X_dev_2)