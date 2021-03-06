#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:50:56 2018

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
from keras.layers import Dense, Dropout 
from keras import initializers
from keras import regularizers 
from sklearn.model_selection import train_test_split

from utils_1 import *

#################### 1 - EXPERIMENT 1 #################### 

########## 1.1 - Loading the data sets 

## loading data
data = sio.loadmat('Cleaned_data.mat', squeeze_me=True)

## 2011 data
X_train_1 = data['X_train_2011']
Y_train_1 = data['Y_train_2011']


## 2015 data
X_2015 = data['X_train_2015']
Y_2015 = data['Y_train_2015']

## Spliting dev and test sets
X_dev_1, X_test_1, Y_dev_1, Y_test_1 = train_test_split(X_2015, Y_2015 ,test_size=0.5,shuffle = False)

########## 1.2 - Shallow Neural Network
# define base model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=242, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros'))
    # Custom optimizer
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)
	# Compile model
    model.compile(loss='mean_absolute_error', optimizer=adam)
    return model

snn = baseline_model()
snn.summary()

history1 = snn.fit(X_train_1, Y_train_1, epochs = 20, batch_size = 1024, shuffle=False)

##### Computing the loss
training_error_1 = snn.evaluate(X_train_1,Y_train_1)
dev_error_1 = snn.evaluate(X_dev_1,Y_dev_1)

##### Generating predicted cleanbottom
Y_pred_train_1_snn = snn.predict(X_train_1)
Y_pred_dev_1_snn = snn.predict(X_dev_1)


## Debugging: the loss is supposed to be non increasing

losses = history1.history["loss"]

epoch = history1.epoch
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.scatter(epoch, losses)
plt.show()


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

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=242, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros'))
    # Custom optimizer
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)
	# Compile model
    model.compile(loss='mean_absolute_error', optimizer=adam)
    return model

snn = baseline_model()
snn.summary()

history = snn.fit(X_train_2, Y_train_2, epochs = 20, batch_size = 1024, shuffle=False)

##### Computing the loss
training_error_2 = snn.evaluate(X_train_2,Y_train_2)
dev_error_2 = snn.evaluate(X_dev_2,Y_dev_2)


##### Generating predicted cleanbottom
Y_pred_train_2_snn = snn.predict(X_train_2)
Y_pred_dev_2_snn = snn.predict(X_dev_2)

## Debugging: the loss is supposed to be non increasing

losses = history2.history["loss"]

epoch = history2.epoch
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.scatter(epoch, losses)
plt.show()
