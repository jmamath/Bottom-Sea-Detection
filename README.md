# Bottom-Sea-Detection



## Context and Objectives
Terabytes of acoustic multispectral data are routinely collected by oceanographic vessel on fisheries resources campaign assessment in West African waters. The vessel sends out pulses of various frequency's acoustic waves in the water, those waves are reflected back to the source when they meet diverse organisms (fish, plankton, etc) or more generally solid objects. We call echogram (echo more informally) the corresponding signal. Since there is a large variety of elements interfering with the acoustic signal it is hard (we rely on the work of experts) to interpret and identify the various organisms that are found by this procedure. Nowadays bottom sea recognition is still being done by the work of experts.
The overall goal of this work is to use machine learning and deep learning methods to automate the task of bottom sea recognition.
## Data
Data has been provided by IRD (Institut de Recherche et Développement), it consists of two files 2011 and 2015 corresponding to two campaigns that took places in those years respectively. We can summarise the relevant information as follows:
After having discussed with the experts' team it seems that the following variables were relevant to learn from data:Latitude, Longitude, Echogram, Depth, and CleanBottom. For our purpose we only select  the lowest frequency (18 kHz) to draw the bottom line (Mainly because it goes deeper).
* Echogram is associated with Depth, in fact for every value of depth there is an echogram.
* Depth is a grid with a regular spacing and each cell correspond to a value of depth in meters
* CleanBottom are the values of the bottom set by the expert.
* Latitude
* Longitude
In summary, data can be viewed as a snapshot of the water where at each time (ping) we have diverse values. Hence we subset our training set using those variables.


## Sampling Methodology
We sampled 4 sets with the same settings, the first example was sampled randomly so that we could select a compact set preserving a continuity condition, (in other words subsequents pings):
* Training_2011
* Training_2015
* Test_2011
* Test_2015 
Each training set have 100,000 examples, and each test set have 30,000 examples.
It is possible to visualize the campaign with the image below. 


Since we are dealing with time series, and that there are 2661063 pings we made a script `RandSelectOne.m`
(that must be present in each folder containing Echogram.mat and Filtering.mat) that randomly select a number between 1 and 2661063 so that you can subset 100000 straight values.
We then selected 100000 examples for a training set and 30000 examples for a test set from both distribution 2011 and 2015.

In summary we have 4 sets :
* Training_2011.mat
* Test_2011.mat
* Training_2015.mat
* Test_2015.mat

![Alt text](Plots/Composition.jpg?raw=false "Training Set")

## Modeling Methodology

### Introduction and Problem Setting
Before starting any modeling we needed to know the kind of problem we may refer on.Clearly, we can frame it as a supervised learning problem, where the variable to predict is CleanBottom that we rename Y, doing so, our explanatory variable will be X a DataFrame with all other data without Depth. In summary
* Y = CleanBottom
* X = Latitude, Longitude, Echogram.

In a first time, we opted for a regression problem, in other words we want to evaluate the value of the bottom Y given X. Our error will be measured in meters.
Hence we tried several models:
* Linear regression (LR)
* Shallow Neural Network (SNN)
* Deep Neural Network (DNN)
* Convolutional Neural Network (CNN)
* Recurrent Neural Network (RNN)

In the following paragraphs we will describe how the data were preprocessed in order to be fed in the models.

### Data cleaning and Preprocessing 
#### Data Cleaning
In the following image we can see how the echogram looks like. The color gradient quantify the response of the pulse at each ping and at each depth. When the response of the sound pulse is very high the value had been changed to NA that is why it seems there is rectangles at the bottom of the image. The red line is the corrected bottom value given by the expert. 

![Alt text](Plots/echo_example.jpg?raw=false "Training Set")

Since our goal is to predict the “red line” using echogram informations here is what was done:
* Removed all echogram value with NA values below the bottom sea. 
* Selected 20 meters of echogram above this limit. Which correspond to 120 cells of echogram for each ping.
* Removed all pings where the corrected bottom were deeper than 500m.

The last action is motivated because the experts are interested only to predict values which are lower than 500 m.

To illustrate what is the output of this data cleaning we give the example of the set Training_2011.  


| Variable name | Size before cleaning        | Size after cleaning           | 
|-------------|:-------------:|:-------------:|
| CleanBottom | (100000,1)     | (79426, 1)| 
| Latitude | (100000,1)     | (79426, 1)     |   
| Longitude | (100000,1)| (79426, 1)     |  
| Echogram | (100000, 2581) | (79426, 120)     |    
|   Depth      | (100000, 2581)   |  (79426, 120)   |   

The goal of this procedure is to focus on relevant data, and to avoid training models for too long. Now let’s describe how the data where stacked to be fed in each model.

#### Preprocessing
Keeping the training_2011 as an illustrative example, here is how we prepared the data for the models different models

##### Linear Regression, Shallow Neural Network, Deep Neural Network
| Name | Size         | Composed of           | 
|-------------|:-------------:|:-------------:|
| X_train| (79426, 242)    | Latitude, Longitude, Echogram, Depth| 
| Y_train | (79426, 1)    | CleanBottom    |   

The variables are simply stacked horizontally.

##### Convolutional Neural Network

| Name | Size         | Composed of           | 
|-------------|:-------------:|:-------------:|
| X_train| (79426, 120, 4)   | Latitude, Longitude, Echogram, Depth| 
| Y_train | (79426, 1)    | CleanBottom    |  

To feed the Convolutional network we instead stack Echogram and Depth horizontally, and we extend the Latitude and Longitude so that it has the same dimension to form a tensor. In some sense the information from Latitude and Longitude is repeated at each ping for 120 cells.

##### Recurrent Neural Letwork

| Name | Size         | Composed of           | 
|-------------|:-------------:|:-------------:|
| X_train| ( 794, 100, 242)  | Latitude, Longitude, Echogram, Depth| 
| Y_train | ( 794, 100, 1)  | CleanBottom    |  

The philosophy behind recurrent network is to provide data as sequence and to learn temporal relationships. But we also need our test set to be in the same format as our training set. 
So from 79,426 examples we drop the 26 last examples to get 79,400 examples. Then we split it into 794 sequences of 100 examples. Each example being a 242 dimensional vector as for the LM. We made that sequence partitioning in order for the test set to be fed in the network.

### Models

The Linear Regression was run using sklearn, All the following neural network were built using keras. 
#### Linear Regression

```
## Creating a LinearRegression object
lm = LinearRegression()
# Measuring time of training
start = time.time()
lm.fit(X_train,Y_train)
end = time.time()
regression_time = end-start

Y_pred_test_lm = lm.predict(X_test)
Y_pred_train_lm = lm.predict(X_train)

# Measuring effectiveness of the linear model, computing training and test error;
# mae stand for Mean Absolute Error 

mae_train = np.mean(abs(Y_train-Y_pred_train_lm))
mae_test = np.mean(abs(Y_test-Y_pred_test_lm))
```

#### Shallow Neural Network

Here we tried a very basic model with one hidden layer and 4 neurons and no regularization.

```
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

# Computing time on training

start = time.time()
history = snn.fit(X_train, Y_train, epochs = 12, batch_size = 1024, shuffle=True)
end = time.time()
SNN_time = end-start


training_error = snn.evaluate(X_train,Y_train)
test_error = snn.evaluate(X_test,Y_test)
Y_pred_train_snn = snn.predict(X_train)
Y_pred_test_snn = snn.predict(X_test)
```

##### SNN Summary
```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 4)                 972       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 5         
=================================================================
Total params: 977
Trainable params: 977
Non-trainable params: 0
_________________________________________________________________
```

#### Deep Neural Network

I called this model DNN but it is just deeper than the first neural network, and I added a regularization function

```
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(240, input_dim=242, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(120, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(60, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(30, kernel_initializer="normal", bias_initializer='zeros'))
    model.add(Dense(15, kernel_initializer="normal", bias_initializer='zeros'))    
    model.add(Dense(7, kernel_initializer="normal", bias_initializer='zeros'))    
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros',kernel_regularizer=regularizers.l1(0.7)))
	# Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

dnn = baseline_model()
dnn.summary()
## Measuring time performance
start = time.time()
history = dnn.fit(X_train, Y_train, epochs = 12, batch_size = 1024, shuffle=True)
end = time.time()
DNN_time = end-start



training_error = dnn.evaluate(X_train,Y_train)
test_error = dnn.evaluate(X_test,Y_test)
Y_pred_train_dnn = dnn.predict(X_train)
Y_pred_test_dnn = dnn.predict(X_test)
```
##### DNN Summary

```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_10 (Dense)             (None, 240)               58320     
_________________________________________________________________
dense_11 (Dense)             (None, 120)               28920     
_________________________________________________________________
dense_12 (Dense)             (None, 60)                7260      
_________________________________________________________________
dense_13 (Dense)             (None, 30)                1830      
_________________________________________________________________
dense_14 (Dense)             (None, 15)                465       
_________________________________________________________________
dense_15 (Dense)             (None, 7)                 112       
_________________________________________________________________
dense_16 (Dense)             (None, 1)                 8         
=================================================================
Total params: 96,915
Trainable params: 96,915
Non-trainable params: 0
_________________________________________________________________
```

#### Convolutional Neural Network
Only one convolutional layer was used followed by a Max Pooling before flatten the result and give it to a fully connected  neural network. Notice that it follow the same parametrization as our SNN model. The goal was to learn with as few parameters as possible.
```
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
    model.add(Dense(1, kernel_initializer="normal", bias_initializer='zeros',kernel_regularizer=regularizers.l1(0.7)))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

cnn = conv_model()
cnn.summary()

## Measuring time performance
start = time.time()
history = cnn.fit(X_train, Y_train, epochs = 26, batch_size = 1024, shuffle=True)
end = time.time()
CNN_time = end-start

training_error = cnn.evaluate(X_train,Y_train)
test_error = cnn.evaluate(X_test,Y_test)
Y_pred_train_cnn = cnn.predict(X_train)
Y_pred_test_cnn = cnn.predict(X_test)
```

##### Convolutional Neural Network Summary

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_3 (Conv1D)            (None, 101, 1)            81        
_________________________________________________________________
batch_normalization_3 (Batch (None, 101, 1)            4         
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 82, 1)             0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 82, 1)             4         
_________________________________________________________________
flatten_2 (Flatten)          (None, 82)                0         
_________________________________________________________________
dense_19 (Dense)             (None, 4)                 332       
_________________________________________________________________
dense_20 (Dense)             (None, 1)                 5         
=================================================================
Total params: 426
Trainable params: 422
Non-trainable params: 4
_________________________________________________________________
```

#### Recurrent Neural Network
We used a bidirectional GRU layer over sequences of 100 examples in order for the network to be sensitive to sudden trenches.
The network take longuer to train, here 200 epoch to have fairly good results.
```
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

## Measuring time performance
start = time.time()
history = rnn.fit(X_train, Y_train, epochs = 200, batch_size = 512, shuffle=False)
end = time.time()
RNN_time = end-start

training_error = cnn.evaluate(X_train,Y_train)
test_error = cnn.evaluate(X_test,Y_test)
Y_pred_train_cnn = cnn.predict(X_train)
Y_pred_test_cnn = cnn.predict(X_test)
```
##### Recurrent Neural Network Summary
```
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_2 (Bidirection (None, 100, 8)            5928      
_________________________________________________________________
time_distributed_6 (TimeDist (None, 100, 4)            36        
_________________________________________________________________
time_distributed_7 (TimeDist (None, 100, 1)            5         
=================================================================
Total params: 5,969
Trainable params: 5,969
Non-trainable params: 0
_________________________________________________________________
```
