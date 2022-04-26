function [xTrain, tTrain, xTest, tTest] = DatasetInit(DatasetName)

% ===== FOR CLASSIFICATION DATASET ONLY ===== %

% Read dataset from files
% Generate Target Class
% Shuffle dataset's order
% Store in variables

% ==================================================

%x2
if DatasetName == "COVID7_WINDOWX2"
    %Load Data from File
    data1 = load('covid7_windowx2.txt');
    %[X,~] = mapminmax(data(:,1:8),0,1);
    data2=data1(:,2:end);
 
    data=data2/3553720;
    %data=mapminmax(data2,0,1); 
    %map1=data(:,1);
          %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = 1:size(data,1);%randperm(size(data,1));
    xTrain = data(I(1:500),1:7);
    tTrain = data(I(1:500),8);
    xTest = data(I(501:end),1:7);
    tTest = data(I(501:end),8);
    clear data I X T;% xTrain tTrain xTest tTest;
end
%----------------------------------------------------

%x3
if DatasetName == "COVID7_WINDOWX3"
    %Load Data from File
    data1 = load('covid7_windowx3.txt');
    %[X,~] = mapminmax(data(:,1:8),0,1);
    data2=data1(:,2:end);
 
    data=data2/3553720;
    %data=mapminmax(data2,0,1); 
    %map1=data(:,1);
          %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = 1:size(data,1);%randperm(size(data,1));
    xTrain = data(I(1:500),1:7);
    tTrain = data(I(1:500),8);
    xTest = data(I(501:end),1:7);
    tTest = data(I(501:end),8);
    clear data I X T;% xTrain tTrain xTest tTest;
end
%----------------------------------------------------
%x4
if DatasetName == "COVID7_WINDOWX4"
    %Load Data from File
    data1 = load('covid7_windowx4.txt');
    %[X,~] = mapminmax(data(:,1:8),0,1);
    data2=data1(:,2:end);
 
    data=data2/3553720;
    %data=mapminmax(data2,0,1); 
    %map1=data(:,1);
          %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = 1:size(data,1);%randperm(size(data,1));
    xTrain = data(I(1:500),1:7);
    tTrain = data(I(1:500),8);
    xTest = data(I(501:end),1:7);
    tTest = data(I(501:end),8);
    clear data I X T;% xTrain tTrain xTest tTest;
end
%----------------------------------------------------
%x5
if DatasetName == "COVID7_WINDOWX5"
    %Load Data from File
    data1 = load('covid7_windowx5.txt');
    %[X,~] = mapminmax(data(:,1:8),0,1);
    data2=data1(:,2:end);
 
    data=data2/3553720;
    %data=mapminmax(data2,0,1); 
    %map1=data(:,1);
          %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = 1:size(data,1);%randperm(size(data,1));
    xTrain = data(I(1:500),1:7);
    tTrain = data(I(1:500),8);
    xTest = data(I(501:end),1:7);
    tTest = data(I(501:end),8);
    clear data I X T;% xTrain tTrain xTest tTest;
end
%----------------------------------------------------
if DatasetName == "COVID_7d"
    %Load Data from File
    data1 = load('covid_new7.txt');
    %[X,~] = mapminmax(data(:,1:8),0,1);
    data2=data1(:,2:end);
 
    data=data2/3553720;
    %data=mapminmax(data2,0,1); 
    %map1=data(:,1);
          %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = 1:size(data,1);%randperm(size(data,1));
    xTrain = data(I(1:500),1:7);
    tTrain = data(I(1:500),8);
    xTest = data(I(501:end),1:7);
    tTest = data(I(501:end),8);
    clear data I X T;% xTrain tTrain xTest tTest;
end

if DatasetName == "COVID_14d"
    %Load Data from File
    data1 = load('covid_new14.txt');
    %[X,~] = mapminmax(data(:,1:8),0,1);
    data2=data1(:,2:end);
 
    data=data2/3553720;
    %data=mapminmax(data2,0,1); 
    %map1=data(:,1);
          %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = 1:size(data,1);%randperm(size(data,1));
    xTrain = data(I(1:500),1:14);
    tTrain = data(I(1:500),15);
    xTest = data(I(501:end),1:14);
    tTest = data(I(501:end),15);
    clear data I X T;% xTrain tTrain xTest tTest;
end

if DatasetName == "COVID_30d"
    %Load Data from File
    data1 = load('covid_new30.txt');
    %[X,~] = mapminmax(data(:,1:8),0,1);
    data2=data1(:,2:end);
 
    data=data2/3553720;
    %data=mapminmax(data2,0,1); 
    %map1=data(:,1);
          %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = 1:size(data,1);%randperm(size(data,1));
    xTrain = data(I(1:500),1:30);
    tTrain = data(I(1:500),31);
    xTest = data(I(501:end),1:30);
    tTest = data(I(501:end),31);
    clear data I X T;% xTrain tTrain xTest tTest;
end


% ==================================================


end