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

The datasets and the procedure to get them is described in the File Data *Matecho_UserManual_18_05_2017.pdf* 
I also add a screen shot of all the variables of each dataset. After having discussed further with the team of experts it seems that the following variables were relevant to learn from data : Time, Latitude, Longitude, Echogram, Depth and CleanBottom. We describe it for the 2011 campaign because the structure of the data are similar in 2015. Echogram and Depth correspond respectively to Echogram18 and Depth18 since we always take the lowest frequency (18 kHz) to draw the bottom line (Mainly because low frequency goes deeper). 
* Echogram is associated with Depth, in fact for every value of depth there is an echogram.
* Depth has 2581 values each spaced by 0.1916m. `min(Depth) = 5.5`, `max(Depth) = 499.86928`
* CleanBottom is the values of the bottom set by the expert.
* Time : numbers of second since January 1st 1970.

In summary data can be viewed as a snapchot of the water where at each time (ping) we have diverses values.
Hence we subset our training set using those variables. 

| Name        | Size           | Class  |
| ------------- |:-------------:| -----:|
| CleanBottom      | (1,100000) | double |
| Depth      | (1,2581)     |   double |
| Latitude | (1,100000)      |  double |
| Longitude | (1,100000)      |   double |
| Echogram | (2581,100000)    |    single |
| Time | (1,100000)     |    double |

Here we select 100000 values, the Matlab script used to subset our training and test set is given in the file *Preprocessing*




## Methodology

## Result
