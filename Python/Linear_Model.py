#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:26:04 2018

@author: root
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



#################### 1 - Preprocessing #################### 

########## 1.1 - Loading the training set and test set

training = sio.loadmat('Training_2015.mat', squeeze_me=True)
test = sio.loadmat('Testset_2011.mat', squeeze_me=True)

# Get variables names, both dataset have the same variables
training.keys()

# Extracting variables
# Depth is the same in both dataset
Depth_train = training['Depth']
Depth_test = test['Depth']

Echogram_train = training['Echogram']
CleanBottom_train = training['CleanBottom']
Longitude_train = training['Longitude']
Latitude_train = training['Latitude']



Echogram_test = test['Echogram']
CleanBottom_test = test['CleanBottom']
Longitude_test = test['Longitude']
Latitude_test = test['Latitude']

########## 1.2 - Getting the index of the last non Nan value
## We start by getting indices where there CleanBottom is deeper than 500
# and when it is not.

### splitting the training CleanBottom
deep_index_train = CleanBottom_train >= 500
deep_index_train = deep_index_train.nonzero()[0]

shallow_index_train = CleanBottom_train < 500
shallow_index_train = shallow_index_train.nonzero()[0]

### splitting the test CleanBottom
deep_index_test = CleanBottom_test >= 500
deep_index_test = deep_index_test.nonzero()[0]

shallow_index_test = CleanBottom_test < 500
shallow_index_test = shallow_index_test.nonzero()[0]

## We create a function to get the last non Nan value
def last_depth_index(Echogram, shallow_index, deep_index):
    """
    In this function we will return the last non nan value of Echogram.
    
    Arguments :
    - Echogram is the dataset of Echogram18
    - shallow_index is the indices of all pings corresponding to a bottom < 500
    - deep_index is all other pings associated with a bottom >= 500
    
    Output :
    A vector echolight wth the same length than CleanBottom but with only 
    the Sv of the deepest non nan value.    
    """
    # 0 - Getting some relevant shape from the arguments
    (n,m) = Echogram.shape
    deepsize = deep_index.size
    shallowsize = shallow_index.size
    
    ## 1 - We create three dataset : lastDepthIndex, deepdepth and shallowdepth 
    ## lastDepthIndex will be for each ping the index of the last non nan value.
    deepdepth = np.zeros(deepsize, dtype = int)
    shallowdepth = np.zeros(shallowsize, dtype = int)
    lastDepthIndex = np.zeros(m, dtype = int)
    
    # the indices associated with deep index does not admit any nan so we turn
    # back the last index 
    deepdepth += (n-1)
    
    ## Here we get the index of the last non nan value to convince yourself that
    # it works look at the section test down below.
    shallowdepth = np.sum(~np.isnan(Echogram[:,shallow_index]), axis=0) -1
    
    ## Now it's time to fill out lastDepthIndex 
    lastDepthIndex[shallow_index] = shallowdepth
    lastDepthIndex[deep_index] = deepdepth
    
    return lastDepthIndex



lastDepthIndex_train = last_depth_index(Echogram_train, shallow_index_train, deep_index_train)
lastDepthIndex_test = last_depth_index(Echogram_test, shallow_index_test, deep_index_test)


########## 1.3 - Getting a lighted version of Echogram

### For the training set
(n_train,m_train) = Echogram_train.shape
columns = np.arange(m_train)
echo_indices = list(zip(lastDepthIndex_train,columns))

echolight_train = np.zeros(m_train)
for kping in range(m_train):
    echolight_train[kping] = Echogram_train[echo_indices[kping]]
    
### For the test set
(n_test,m_test) = Echogram_test.shape
columns = np.arange(m_test)
echo_indices = list(zip(lastDepthIndex_test,columns))

echolight_test = np.zeros(m_test)
for kping in range(m_test):
    echolight_test[kping] = Echogram_test[echo_indices[kping]]
    
    
# Ok now we are ready to fit any model

#################### 2 - Linear Regression #################### 
# Since we do not have the echogram for corresponding to depth greater than 
# 500m we set all of bottom values found greater tha 500m to 500m  

CleanBottom_train[deep_index_train] = 500    
CleanBottom_test[deep_index_test] = 500

## Then we select for each ping the depth corresponding to 
# the last non Nan value in Echogram
pingDepth_train = np.zeros(m_train)
for kping in range(m_train):
    pingDepth_train[kping] = Depth_train[lastDepthIndex_train[kping]]
    
pingDepth_test = np.zeros(m_test)
for kping in range(m_test):
    pingDepth_test[kping] = Depth_test[lastDepthIndex_test[kping]]    
    
    
    
# Now let's create some DataFrame to fit the model  

## Creating X_train and Y_train        
pingDepth_train = pd.DataFrame(pingDepth_train, columns = ['pingDepth'])  
echoLight_train = pd.DataFrame(echolight_train, columns = ['echoLight'])
latitude_train = pd.DataFrame(Latitude_train, columns = ['latitude'])
longitude_train = pd.DataFrame(Longitude_train, columns = ['longitude'])
X_train = pd.concat([pingDepth_train,longitude_train,latitude_train,echoLight_train], axis = 1)
Y_train = pd.DataFrame(CleanBottom_train, columns = ['cleanBottom'])    

## Creating X_test and Y_test
pingDepth_test = pd.DataFrame(pingDepth_test, columns = ['pingDepth'])  
echoLight_test = pd.DataFrame(echolight_test, columns = ['echoLight'])
latitude_test = pd.DataFrame(Latitude_test, columns = ['latitude'])
longitude_test = pd.DataFrame(Longitude_test, columns = ['longitude'])
X_test = pd.concat([pingDepth_test,longitude_test,latitude_test,echoLight_test], axis = 1)
Y_test = pd.DataFrame(CleanBottom_test, columns = ['cleanBottom'])  


## Now let's plot and compare the data.
## As you can see pingDepth and Y looks a lot similar
# Let's plot both data
## Data from the training set
pings = np.arange(m_train)
plt.xlabel('Pings')
plt.ylabel('Profondeur')
plt.plot(pings,pingDepth_train)
plt.plot(pings,Y_train)
plt.legend(['PingDepth', 'CleanBottom']) 
plt.title('Training Set 2015')

## Data from the test set
pings = np.arange(m_test)
plt.xlabel('Pings')
plt.ylabel('Profondeur')
plt.plot(pings,pingDepth_test)
plt.plot(pings,Y_test)
plt.legend(['PingDepth', 'CleanBottom'], loc='lower right') 
plt.title('Test Set 2011')



## Creating a LinearRegression object
lm = LinearRegression()
lm.fit(X_train,Y_train)

Y_pred_test = lm.predict(X_test)
Y_pred_train = lm.predict(X_train)

# Measuring effectiveness of the linear model, computing training and test error;
# mae stand for Mean Absolute Error 

mae_train = np.mean(abs(Y_train-Y_pred_train))
mae_test = np.mean(abs(Y_test-Y_pred_test))
