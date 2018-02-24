# Bottom-Sea-Detection

## Context and Objectives
Terabytes of acoustic multispectral data are routinely collected by oceanographic vessel on campaign to assess fisheries resources in West African waters. The vessel send out pulses of various frequency's acoustic waves in the water, those waves are reflected back to the source when they meet diverse organisms (fish, plancton, etc) or more generaly solid objects. We call echogram (echo more informally) the corresponding signal. Since there is a large variety of elements interfering with the acoustic signal it is hard (we rely on the work of experts) to interpret and indentify the various organisms that are found by this procedure. Nowadays bottom sea recognition is still being done by the work of experts.
The overall goal of this work is to use machine learning and deep learning methods to automate the task of bottom sea recognition.

## Data
Data has been provided by IRD (Institut de Recherche et Développement), it consists of two files 2011 and 2015 corresponding to two campaigns that took places in those years respectively. We can summarise the relevant information as follows:
* 2011
  * Echogram.mat ~ 28.26 GB
  * Filtering.mat ~ 34.5 GB 
* 2015
  * Echogram.mat ~ 30.45 GB
  * Filtering.mat ~ 47.12 GB

The datasets and the procedure to get them is described in here /Data/Matecho_UserManual_18_05_2017.pdf. 
A screen shot of the variables and a descriptive file are present in:
* Data/2011 - three files Echogram_variables.jpg, Filtering_variables.jpg and Parameters_2011.txt
* Data/2015 - three files Echogram_variables.jpg, Filtering_variables.jpg and Parameters_2015.txt
 After having discussed with the experts team it seems that the following variables were relevant to learn from data : Time, Latitude, Longitude, Echogram, Depth and CleanBottom. Echogram and Depth correspond respectively to Echogram18 and Depth18 since we always take the lowest frequency (18 kHz) to draw the bottom line (Mainly because low frequency goes deeper). 
* Echogram is associated with Depth, in fact for every value of depth there is an echogram.
* Depth is a grid with a regular spacing and each cell correspond to a value of depth in metres
* CleanBottom are the values of the bottom set by the expert.
* Time : numbers of second since January 1st 1970.

In summary data can be viewed as a snapchot of the water where at each time (ping) we have diverses values.
Hence we subset our training set using those variables. 

### Subsetting Methodology
Data has been given within a hard disk drive (HDD), so the following is made (for reproductibility purposes) to help someone with the same disk getting the same results.
Here we have selected 100000 values, we provide Matlab scripts to subset a training set and a test set, both can be found in the folder Matlab. The interface for the user are the scripts `Get_TrainingSet.m` and `Get_TestSet.m`. There, one can change how many examples he wants (less or more than 100000). When executed the scripts output the dataset Training.mat and Test.mat in the root directory of the HDD. Here is how it looks:

| Name        | Size           | Class  |
| ------------- |:-------------:| -----:|
| CleanBottom      | (1,100000) | double |
| Depth      | (1,2581)     |   double |
| Latitude | (1,100000)      |  double |
| Longitude | (1,100000)      |   double |
| Echogram | (2581,100000)    |    single |
| Time | (1,100000)     |    double |

Since we are dealing with time series, and that there is 2661063 pings we made a script `RandSelectOne.m`
(that must be present in each folder containing Echogram.mat and Filtering.mat) that randomly select a number between 1 and 2661063 so that you can subset 100000 straigt values.
We then selected 100000 examples for a training set and 30000 examples for a test set from both distribution 2011 and 2015.

In summary we have 4 sets :
* Training_2011.mat
* Test_2011.mat
* Training_2015.mat
* Test_2015.mat

## Modeling Methodology

### Feature Selection and Problem Settings
Before starting any modeling we needed to know the kind of problem we may refer on.Clearly we can frame it as a supervised learning problem, where the variable to predict is CleanBottom that we rename Y, doing so, our explanatory variable will be X a DataFrame with all other data without Depth. In summary
* Y = CleanBottom 
* X = Latitude, Longitude, Echogram, Time.

We want our settings to be general enought to be applied with the data of the 2015 campaign. At first we wanted to treat it as a multi class classification problem with 2581 different classes. The problem with this approach is that it would not help us apply our model to data from a different distribution, in fact the 2015 campaing as a different Depth structure, the spacing between the different values is different than 0.1916, and has different total values. Hence, in a first time we opted for a regression problem. Then our error would be measure in metres.

### Linear Model
Following Occam's razor principle, we opted to fit simple linear model to see how much we could explain just with linear assumptions.
#### Preprocessing
The experts are not interested with bottom values greater than 500m. CleanBottom have values greater than 500m that had been found with a complete version of Echogram, but the Echogram we got was sharped to the first 500m for data volume issues.
Then we set all values of CleanBottom greater or equal than 500m to 500.
There was still a problem, in fact Echogram had too much data and had a lot of Nan values. What a problem at first revealed itself to be an opportunity. After some discussion with the experts, the origin of the Nan values was clear. In fact for example if we take an arbitrary ping and that the bottom is at 100m depth, all the values deeper in echogram will be Nan. We then had a glimpse of the bottom values from Echogram just with the Nan distribution.
Thus we created a new variable that gaves at each ping the depth of the last non-Nan value we call it PingDepth.
After some plotting it seemed that both CleanBottom and PingDepth had the same structure.

![Alt text](Plots/Training_2011.png?raw=false "Training Set")
![Alt text](Plots/Test_2011.png?raw=false "Training Set")


![Alt text](Plots/Training_2015.png?raw=false "Training Set")
![Alt text](Plots/Test_2015.png?raw=false "Training Set")

We also changed Echogram to get Echolight, a lighted version of Echogram, for each ping we select only the value of the last non Nan echo. In summary 
* X = PingDepth, Echolight, Latitude, Longitude
* Y = CleanBottom (with values greater than 500m set to 500m)

We removed Time because it has some Nan values.
Then we fitted the model using sklearn a machine learning module in Python.
The Mean Absolute Error was 5.86, not bad.


