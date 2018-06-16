#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:21:28 2018

@author: root
"""


#################### 0 - IMPORT RELEVANT LIBRARIES ####################

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LinearRegression
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


########## 1.2 Linear regression 
## Creating a LinearRegression object
lm = LinearRegression()

## Fitting the linear regression
lm.fit(X_train_1,Y_train_1)

Y_pred_dev_1_lm = lm.predict(X_dev_1)
Y_pred_train_1_lm = lm.predict(X_train_1)

# Measuring effectiveness of the linear model, computing training and test error;
# mae stand for Mean Absolute Error 
mae_train_1 = np.mean(abs(Y_train_1-Y_pred_train_1_lm))
mae_dev_1 = np.mean(abs(Y_dev_1-Y_pred_dev_1_lm))

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

########## 2.2 Linear regression 
## Creating a LinearRegression object
lm = LinearRegression()

## Fitting the linear regression
lm.fit(X_train_2,Y_train_2)

Y_pred_train_2_lm = lm.predict(X_train_2)
Y_pred_dev_2_lm = lm.predict(X_dev_2)

# Measuring effectiveness of the linear model, computing training and test error;
# mae stand for Mean Absolute Error 
mae_train_2 = np.mean(abs(Y_train_2-Y_pred_train_2_lm))
mae_dev_2 = np.mean(abs(Y_dev_2-Y_pred_dev_2_lm))