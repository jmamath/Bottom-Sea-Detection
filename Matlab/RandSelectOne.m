function RandomIndex = RandSelectOne(no_ping,n,seed)
rng(seed);
% n is the number of pings we want to select
% random number generation between 1 and no_ping - n
startselection = randi(no_ping-n);
endselection = startselection + (n-1);
% effective index selection of the data A
RandomIndex = startselection:endselection;
end
