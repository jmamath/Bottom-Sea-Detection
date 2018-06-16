#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:03:43 2018

@author: root
"""

#################### 0 - IMPORT RELEVANT LIBRARIES ####################

import random as rd
import scipy.io as sio
from keras.layers import Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, LSTM
from keras.models import Model
from keras.layers import TimeDistributed
from keras.layers import Dense, Dropout, Activation 
from keras import initializers
from keras import regularizers 

import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

########## 1.2 - Recurrent Neural Network 
########## 1.2.1 -  Data Reshaping and Preprocessing
## Normalizing inputs
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_1 = scaler.fit_transform(X_train_1)
Y_train_1 = scaler.fit_transform(Y_train_1.reshape(Y_train_1.shape[0],1))

X_dev_1 = scaler.fit_transform(X_dev_1)
Y_dev_1 = scaler.fit_transform(Y_dev_1.reshape(Y_dev_1.shape[0],1))


## Get some reshaping before feed the recurrent model
## Here it's a case by case treatment. because we have to set up time steps.
## And times steps have to be a multiple of batch size


#### 2011
X_train_1 = X_train_1[0:79400,:]
Y_train_1 = Y_train_1[0:79400,:]

X_dev_1 = X_dev_1[0:49900,:]
Y_dev_1 = Y_dev_1[0:49900,:]




### Reshaping
m_train,n = X_train_1.shape
m_dev,n = X_dev_1.shape

pgcd = math.gcd(m_train,m_dev)
batch_train = m_train // pgcd
batch_dev = m_dev // pgcd

X_train_1 = X_train_1.reshape(batch_train,pgcd,n)
Y_train_1 = Y_train_1.reshape(batch_train,pgcd,1)

X_dev_1 = X_dev_1.reshape(batch_dev,pgcd,n)
Y_dev_1 = Y_dev_1.reshape(batch_dev,pgcd,1)


########## 1.2.1 -  Model

# define model
def model():
    # create model
    model = Sequential()        
    model.add(Bidirectional(GRU(4, return_sequences=True), input_shape=[pgcd,n]))
    model.add(TimeDistributed(Dense(4, kernel_initializer="normal", bias_initializer='zeros', activation='tanh')))
    model.add(TimeDistributed(Dense(1, kernel_initializer="normal", bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01))))     
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
rnn = model()
rnn.summary()

history = rnn.fit(X_train_1, Y_train_1, epochs = 50, batch_size = 512, shuffle=False)



## Debugging
losses = history.history["loss"]

epoch = history.epoch
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.scatter(epoch, losses)
plt.show()



# make predictions
Y_pred_train_1_rnn = rnn.predict(X_train_1)
Y_pred_dev_1_rnn = rnn.predict(X_dev_1)



## Reshaping
Y_train_1 = Y_train_1.reshape(m_train,1)
Y_pred_train_1_rnn = Y_pred_train_1_rnn.reshape(m_train,1)
Y_dev_1 = Y_dev_1.reshape(m_dev,1)
Y_pred_dev_1_rnn = Y_pred_dev_1_rnn.reshape(m_dev,1) 



# invert predictions
Y_pred_train_1_rnn = scaler.inverse_transform(Y_pred_train_1_rnn)
Y_train_1 = scaler.inverse_transform(Y_train_1)
Y_pred_dev_1_rnn = scaler.inverse_transform(Y_pred_dev_1_rnn)
Y_dev_1 = scaler.inverse_transform(Y_dev_1)




## Compute the cost
mae_train_1 = np.mean(abs(Y_train_1-Y_pred_train_1_rnn))
mae_dev_1 = np.mean(abs(Y_dev_1-Y_pred_dev_1_rnn))


########## 1.2.2 - Generalization error
## Scale the test set between 0 and 1
X_test_1 = scaler.fit_transform(X_test_1)
Y_test_1 = scaler.fit_transform(Y_test_1.reshape(Y_test_1.shape[0],1))
## Formating to feed the rnn
X_test_1 = X_test_1[0:49900,:]
Y_test_1 = Y_test_1[0:49900,:]
## Reshaping
X_test_1 = X_test_1.reshape(batch_dev,pgcd,n)
Y_test_1 = Y_test_1.reshape(batch_dev,pgcd,1)
### Make prediction
Y_pred_test_1_rnn = rnn.predict(X_test_1)
### Reshaping
Y_test_1 = Y_test_1.reshape(m_dev,1)
Y_pred_test_1_rnn = Y_pred_test_1_rnn.reshape(m_dev,1) 
### Invert prediction
Y_pred_test_1_rnn = scaler.inverse_transform(Y_pred_test_1_rnn)
Y_test_1 = scaler.inverse_transform(Y_test_1)
### Compute the cost
mae_test_1 = np.mean(abs(Y_test_1-Y_pred_test_1_rnn))

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

########## 2.2 - Recurrent Neural Network 
########## 2.2.1 -  Data Reshaping and Preprocessing
## Normalizing inputs
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_2 = scaler.fit_transform(X_train_2)
Y_train_2 = scaler.fit_transform(Y_train_2.reshape(Y_train_2.shape[0],1))

X_dev_2 = scaler.fit_transform(X_dev_2)
Y_dev_2 = scaler.fit_transform(Y_dev_2.reshape(Y_dev_2.shape[0],1))


## Get some reshaping before feed the recurrent model
## Here it's a case by case treatment. because we have to set up time steps.
## And times steps have to be a multiple of batch size

X_train_2 = X_train_2[0:99900,:]
Y_train_2 = Y_train_2[0:99900,:]

X_dev_2 = X_dev_2[0:39700,:]
Y_dev_2 = Y_dev_2[0:39700,:]




### Reshaping
m_train,n = X_train_2.shape
m_dev,n = X_dev_2.shape

pgcd = math.gcd(m_train,m_dev)
batch_train = m_train // pgcd
batch_dev = m_dev // pgcd

X_train_2 = X_train_2.reshape(batch_train,pgcd,n)
Y_train_2 = Y_train_2.reshape(batch_train,pgcd,1)

X_dev_2 = X_dev_2.reshape(batch_dev,pgcd,n)
Y_dev_2 = Y_dev_2.reshape(batch_dev,pgcd,1)



########## 2.2.2 -  Model

# define model
def model():
    # create model
    model = Sequential()        
    model.add(Bidirectional(GRU(4, return_sequences=True), input_shape=[pgcd,n]))
    model.add(TimeDistributed(Dense(4, kernel_initializer="normal", bias_initializer='zeros', activation='tanh')))
    model.add(TimeDistributed(Dense(1, kernel_initializer="normal", bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01))))     
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
rnn = model()
rnn.summary()

history = rnn.fit(X_train_2, Y_train_2, epochs = 150, batch_size = 512, shuffle=False)



## Debugging
losses = history.history["loss"]

epoch = history.epoch
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.scatter(epoch, losses)
plt.show()



# make predictions
Y_pred_train_2_rnn = rnn.predict(X_train_2)
Y_pred_dev_2_rnn = rnn.predict(X_dev_2)

## Reshaping
Y_train_2 = Y_train_2.reshape(m_train,1)
Y_pred_train_2_rnn = Y_pred_train_2_rnn.reshape(m_train,1)
Y_dev_2 = Y_dev_2.reshape(m_dev,1)
Y_pred_dev_2_rnn = Y_pred_dev_2_rnn.reshape(m_dev,1) 

# invert predictions
Y_pred_train_2_rnn = scaler.inverse_transform(Y_pred_train_2_rnn)
Y_train_2 = scaler.inverse_transform(Y_train_2)
Y_pred_dev_2_rnn = scaler.inverse_transform(Y_pred_dev_2_rnn)
Y_dev_2 = scaler.inverse_transform(Y_dev_2)

## Compute the cost
mae_train_2 = np.mean(abs(Y_train_2-Y_pred_train_2_rnn))
mae_dev_2 = np.mean(abs(Y_dev_2-Y_pred_dev_2_rnn))

########## 1.2.2 - Generalization error
## Scale the test set between 0 and 1
X_test_2 = scaler.fit_transform(X_test_2)
Y_test_2 = scaler.fit_transform(Y_test_2.reshape(Y_test_2.shape[0],1))
## Formating to feed the rnn
X_test_2 = X_test_2[0:39700,:]
Y_test_2 = Y_test_2[0:39700,:]
## Reshaping
X_test_2 = X_test_2.reshape(batch_dev,pgcd,n)
Y_test_2 = Y_test_2.reshape(batch_dev,pgcd,1)
### Make prediction
Y_pred_test_2_rnn = rnn.predict(X_test_2)
### Reshaping
Y_test_2 = Y_test_2.reshape(m_dev,1)
Y_pred_test_2_rnn = Y_pred_test_2_rnn.reshape(m_dev,1) 
### Invert prediction
Y_pred_test_2_rnn = scaler.inverse_transform(Y_pred_test_2_rnn)
Y_test_2 = scaler.inverse_transform(Y_test_2)
### Compute the cost
mae_test_2 = np.mean(abs(Y_test_2-Y_pred_test_2_rnn))
