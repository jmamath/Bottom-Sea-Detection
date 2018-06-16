# OVERVIEW

This interface is meant to apply predictive models
on multispectral oceanographic data.
The recommended software are python 3.6 and the Anaconda suite.
Tensorflow must be installed.

The context is described in the document **Bottom Sea Estimation with Deep Learning Methods.pdf**
Note that this work is still in progress, and is planned to be finished at the end of July 2018.
For example the scripts relative to the preprocessing have not been included, only the models.

## MODELS
Five models can be used as scripts:

- Linear_regression.py : linear regression
- ShallowNN.py : shallow neural network
- DeepNN.py : Deep neural network
- ConvNN.py : convolutional neural network
- RecurrentNN.py : Recurrent neural network
- VariableNN.py : One hidden layer neural network with automatically tuned hyperparameters

## DATA
The data is already preprocessed, two data files are available:
- Cleaned_data.mat
- Conv_data.mat

The first data are in a format to be fed in Linear_Model, ShallowNN, DeepNN, and RecurrentNet. Only the ConvNet need another format, hence Conv_data is right
for it.

## HOW TO USE THE SCRIPTS?
All the scripts share the same structure. They are divided into three subpart

0 - Import the relevant libraries
1 - Load datasets
2 - Model

Among those parts, the section Model is worth some comments:
The models are run with 2011 data and 2015 data
in separate sections.

Keras neural networks can be roughly described in two steps.
- First, we implement the graph structure.
- Second, we run the learning algorithm.

When history has finished running, it's possible to draw some predictions
and to measure training and test error.
The losses are plotted per epoch to see if the learning has taken place. The trend should be to decrease.

Side note, the recurrent network requires a bit of shape manipulation.
That's why it's script is slightly different.
