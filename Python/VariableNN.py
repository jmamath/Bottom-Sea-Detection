#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:07:39 2018

@author: jmamath
"""

#################### 0 - IMPORT RELEVANT LIBRARIES ####################

import scipy.io as sio
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import GPyOpt

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras import initializers
from keras import optimizers 
from keras import regularizers 
from sklearn.model_selection import train_test_split


#################### 1 - HYPERPARAMETERS TUNING WITH BAYESIAN OPTIMISATION ####################

########## 1.1 - Loading the data sets 

## loading data
data = sio.loadmat('Cleaned_data.mat', squeeze_me=True)

## 2011 data
X_train_1 = np.array(data['X_train_2011'])
Y_train_1 = np.array(data['Y_train_2011'])

m = len(Y_train_1)
Y_train_1 = Y_train_1.reshape(m,1)

########## 1.2 - Shallow Neural Network
# define base model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=242, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros'))
	# Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

snn = baseline_model()
snn.summary()


baseline = snn.fit(X_train_1, Y_train_1,epochs = 5, batch_size = 1024)
baseline_score = snn.evaluate(X_train_1, Y_train_1)

# Model to optimize
def variable_neural_network_model(learning_rate,no_unit_layer1):
    # create model
    model = Sequential()
    model.add(Dense(no_unit_layer1, input_dim=242, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros'))
    # Custom optimizer
    adam = optimizers.Adam(lr=learning_rate)
	# Compile model
    model.compile(loss='mean_absolute_error', optimizer=adam)
    return model


# Function to optimize
# We aim to maximize this function with respect to the parameters
# learning_rate and no_unit_layer1
def f(x):
    learning_rate = float(x[:,0])
    no_unit_layer1 = int(x[:,1])
    vnn = variable_neural_network_model(learning_rate,no_unit_layer1)
    history = vnn.fit(X_train_1, Y_train_1,epochs = 5, batch_size = 1024)
    score = vnn.evaluate(X_train_1, Y_train_1)
    return score

bounds = [
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (10**-4,10**-2)},
            {'name': 'no_unit_layer1', 'type': 'discrete', 'domain': np.arange(5,100)}
         ]
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds,
                                                        acquisition_type = 'MPI',
                                                        acquisition_par = 0.1,                                                    
                                                        model_type='GP')

optimizer.run_optimization(max_iter=30, eps=-1)

# Measuring the performance improvement
performance_boost_30_run = (baseline_score/np.min(optimizer.Y) -1)*100


