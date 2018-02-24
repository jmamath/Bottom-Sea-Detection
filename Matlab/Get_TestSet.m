% To subset our test set. 
% The only difference with the training set is that we changed the seed.

cd('/Volumes/AWA_bck/2011/ConvertedHac/Cruise_2011/MatData')
data = matfile('Echogram.mat');

no_select = 30000;     % Numbers of examples to subset
seed = 2;    % the seed plays a role to select a random start and then subset no_select
                % examples from that.
m = length(data.Time);      % Total number of examples in the dataset
RandomIndex = RandSelectOne(m,no_select,seed);  
Latitude = data.Latitude(1,RandomIndex);
Longitude = data.Longitude(1,RandomIndex);
Time = data.Time(1,RandomIndex);
Echogram = data.Echogram18(:,RandomIndex);
Depth = data.Depth18;

cd('/Volumes/AWA_bck/2011/ConvertedHac/Cruise_2011/Treatment2017/DataFilterResults')
data = matfile('Filtering.mat');
CleanBottom = data.CleanBottom(1,RandomIndex);

cd('/Volumes/AWA_bck')
save 'Testset.mat' Longitude Latitude Time Echogram Depth CleanBottom -v7
