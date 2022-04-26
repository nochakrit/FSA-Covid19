%% Credits %%
%  Traning Feed-forward Neural Networks using Grey Wolf Optimizer   %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili,How effective is the Grey Wolf         %
%               optimizer in training multi-layer perceptrons       %
%              Applied Intelligece, in press, 2015,                 %
%               http://dx.doi.org/10.1007/s10489-014-0645-7         %
%                                                                   %

% This function containts full information and implementations of the
% datasets

% lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
% up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]
% dim is the number of variables (dimension of the problem)
%%

function [lb,ub,dim,fobj,inp,hidn,outp] = GetFunctionsInfo(F,NoOfHidden)

switch F
        
        case 'F1'
        fobj = @COVID_7d;
        lb=-10;
        ub=10;
        %dim=46;
        inp=7;
        hidn=NoOfHidden;
        outp=1;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
        case 'F2'
        fobj = @COVID_14d;
        lb=-10;
        ub=10;
        %dim=46;
        inp=14;
        hidn=NoOfHidden;
        outp=1;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
        case 'F3'
        fobj = @COVID_30d;
        lb=-10;
        ub=10;
        %dim=46;
        inp=30;
        hidn=NoOfHidden;
        outp=1;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
        case 'F4'
        fobj = @COVID7_WINDOWX2;
        lb=-10;
        ub=10;
        %dim=46;
        inp=7;
        hidn=NoOfHidden;
        outp=1;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
        case 'F5'
        fobj = @COVID7_WINDOWX3;
        lb=-10;
        ub=10;
        %dim=46;
        inp=7;
        hidn=NoOfHidden;
        outp=1;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
        case 'F6'
        fobj = @COVID7_WINDOWX4;
        lb=-10;
        ub=10;
        %dim=46;
        inp=7;
        hidn=NoOfHidden;
        outp=1;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
        case 'F7'
        fobj = @COVID7_WINDOWX5;
        lb=-10;
        ub=10;
        %dim=46;
        inp=7;
        hidn=NoOfHidden;
        outp=1;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
end

end

%x2
function o=COVID7_WINDOWX2(solution,mlpConfig)
[xTrain, tTrain, ~, ~] = DatasetInit("COVID7_WINDOWX2");


%[xTrain,~] = mapminmax(xTrain,0,1);
%[tTrain,~]= mapminmax(tTrain,0,1);

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight
fitness = 0;
[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
end
fitness = mse(e);
o=fitness;
end
%-------------------------------------------------------

%x2
function o=COVID7_WINDOWX3(solution,mlpConfig)
[xTrain, tTrain, ~, ~] = DatasetInit("COVID7_WINDOWX3");


%[xTrain,~] = mapminmax(xTrain,0,1);
%[tTrain,~]= mapminmax(tTrain,0,1);

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight
fitness = 0;
[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
end
fitness = mse(e);
o=fitness;
end
%-------------------------------------------------------

%x2
function o=COVID7_WINDOWX4(solution,mlpConfig)
[xTrain, tTrain, ~, ~] = DatasetInit("COVID7_WINDOWX4");


%[xTrain,~] = mapminmax(xTrain,0,1);
%[tTrain,~]= mapminmax(tTrain,0,1);

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight
fitness = 0;
[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
end
fitness = mse(e);
o=fitness;
end
%-------------------------------------------------------

%x2
function o=COVID7_WINDOWX5(solution,mlpConfig)
[xTrain, tTrain, ~, ~] = DatasetInit("COVID7_WINDOWX5");


%[xTrain,~] = mapminmax(xTrain,0,1);
%[tTrain,~]= mapminmax(tTrain,0,1);

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight
fitness = 0;
[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
end
fitness = mse(e);
o=fitness;
end
%-------------------------------------------------------

function o=COVID_7d(solution,mlpConfig)
[xTrain, tTrain, ~, ~] = DatasetInit("COVID_7d");


%[xTrain,~] = mapminmax(xTrain,0,1);
%[tTrain,~]= mapminmax(tTrain,0,1);

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight
fitness = 0;
[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
end
fitness = mse(e);
o=fitness;
end

function o=COVID_14d(solution,mlpConfig)
[xTrain, tTrain, ~, ~] = DatasetInit("COVID_14d");


%[xTrain,~] = mapminmax(xTrain,0,1);
%[tTrain,~]= mapminmax(tTrain,0,1);

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
    
end

fitness = mse(e);
o=fitness;

end

function o=COVID_30d(solution,mlpConfig)
[xTrain, tTrain, ~, ~] = DatasetInit("COVID_30d");


%[xTrain,~] = mapminmax(xTrain,0,1);
%[tTrain,~]= mapminmax(tTrain,0,1);

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
    
end

fitness = mse(e);
o=fitness;

end
