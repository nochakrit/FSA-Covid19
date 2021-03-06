%%  Training Feed-forward Neural Networks using Optimizer %%

clear all;
clc;

%% Dataset No. %%
% classification datasets %

% Function_name='F1'; %MLP_XOR dataset
% Function_name='F2'; %MLP_Baloon dataset
% Function_name='F3'; %MLP_Iris dataset
% Function_name='F4'; %MLP_Cancer dataset
% Function_name='F5'; %MLP_Heart dataset

% Function approximation datasets %

% Function_name='F6'; %MLP_Sigmoid dataset
% Function_name='F7'; %MLP_Cosine dataset
% Function_name='F8'; %MLP_Sine dataset

% high-dimensional classification datasets %

% Function_name='F9'; %MLP_Glass dataset
% Function_name='F10'; %MLP_Libras dataset
% Function_name='F11'; %MLP_Thyroid dataset
% Function_name='F12'; %MLP_Wine dataset
% Function_name='F13'; %MLP_Gait dataset

%DatasetName = {'XOR';'Balloon';'Iris';'Cancer';'Heart';'Sigmoid';'Cosine';'Sine';'Glass';'Libras';'Thyroid';'Wine';'Gait'};
%OptimizerName = {'MLP_GWO';'MLP_SMA';'MLP_GBO';'MLP_ACO';'MLP_WOA';'MLP_HPSOGWO';'MLP_GHPSOGWO';'MLP_ADAMHPSOGWO'};
%OptimizerName = {'MLP_PSO';'MLP_GWO';'MLP_IGWO';'MLP_HPSOGWO';'MLP_GHPSOGWO';'MLP_ADAMHPSOGWO';'MLP_SMA';'MLP_GBO';'MLP_WOA'};

%DatasetName = {'XOR';'Balloon';'Iris';'Cancer';'Heart';'Sigmoid';'Cosine';'Sine'};
%OptimizerName = {'MLP_PSO';'MLP_GWO';'MLP_IGWO';'MLP_HPSOGWO';'MLP_GHPSOGWO';'MLP_ADAMHPSOGWO';'MLP_FODHPSOGWO'};

DatasetName = {'XOR';'Balloon';'Iris';'Cancer';'Heart';'Sigmoid';'Cosine';'Sine';'Glass';'Libras';'Thyroid';'Wine';'Gait';'COVID';'COVID_UNDER';'COVID_OVER'};
% OptimizerName = {'MLP_PSO';'MLP_GWO';'AVOA'};
OptimizerName = {'MLP_PSO';'MLP_GWO';'AVOA';'MLP_HPSOGWO';'MLP_GHPSOGWO';'MLP_FODHPSOGWO';'MLP_FODHPSOGWO';'JSOA'};

%% Parameters Configuration %%
RunNo=1;                                   % Max Run
SearchAgentsNo=30;                         % Number of search agents
MaxIteration=10;                           % Maximum number of iterations / SearchAgents
Findex='null';                            % Function Index

for DatasetNo = 14:14
    
    % Dataset
    CurrentDataset = string(DatasetName(DatasetNo));
    disp(strcat('Working on..  ',CurrentDataset,' Dataset'));
    %fprintf('\n');
    
    for OpimizerNo = 1:size(OptimizerName,1)
        
        % Optimizer
        CurrentOptimizer = string(OptimizerName(OpimizerNo));
        disp(strcat(string(OptimizerName(OpimizerNo)),' is Running...'));
        
        % Change Number of Hidden Node
        for HiddenNode = 3
            
            
            % Load details of the selected dataset.
            [lb,ub,dim,fobj,inp,hidn,outp] = GetFunctionsInfo(['F' num2str(DatasetNo)],HiddenNode);
            
            % Parameters for MLP
            mlpConfig.inp = inp;
            mlpConfig.hidn = hidn;
            mlpConfig.outp = outp;
            
            % Population size 50 for XOR and Balloon, 200 for the rest
            %if DatasetNo == 1 || DatasetNo == 2
            %    SearchAgentsNo=50;                         % Customize number of search agents for some datasets
            %end
            
            parfor run = 1:RunNo
                
                watchRun = tic;                       % Elapsed time for each run.
                
                %disp(['>> ' OptimizerAcro ' Run No.' num2str(run) ' is running']);
                
                if OpimizerNo == 1
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve1(run,:)] = PSO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 2
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve2(run,:)] = GWO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 3
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve3(run,:)] = AVOA(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                %if OpimizerNo == 3
                %    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve(run,:)] = IGWO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                %end
                if OpimizerNo == 4
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve4(run,:)] = HPSOGWO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 5
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve5(run,:)] = GHPSOGWO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 7
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve7(run,:)] = ADAMHPSOGWO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 6
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve6(run,:)] = FODHPSOGWO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 8
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurve8(run,:)] = JSOA(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                
                elapsedRun = toc(watchRun);
                ElapsedTimeRun(run,:) = elapsedRun;     % Elapsed time for each run.
                disp(strcat(' >>  ',CurrentOptimizer,' Run No.',num2str(run),' --> ',num2str(HiddenNode),' Hidden Node is done. (',num2str(elapsedRun),' s.)'));
                
            end
            
            
            % Create output directories for each optimizer.
            %DirTimeStamp = datetime('now','TimeZone','Asia/Bangkok','Format','d-MM-y');
            %OutputDir = strcat('Results\',DirTimeStamp);
            OutputDir = strcat('Results\latest');
            if ~exist(OutputDir, 'dir')
                mkdir(OutputDir);
            end
            
            % Save to file.
%             filename = strcat('Results\latest\',CurrentDataset,'_',CurrentOptimizer,'_',num2str(HiddenNode),'_HiddenNode','_Weight_DATA.mat');
%             save(filename,'BestPosition','BestScore','ConvergenceCurve');
%             
%             filename = strcat('Results\latest\',CurrentDataset,'_',CurrentOptimizer,'_',num2str(HiddenNode),'_HiddenNode','_ElapsedTime_DATA.mat');
%             save(filename,'ElapsedTimeRun');
            
            % Testing rate
            [ClassificationRate(OpimizerNo,HiddenNode), ApproximationError(OpimizerNo,HiddenNode)] = TestFitness(['F' num2str(DatasetNo)],RunNo,mlpConfig,BestPosition);
            
            clear BestPosition BestScore ConvergenceCurve ElapsedTimeRun;
            
        end
        
    end
    
    % Save to file.
%     filename = strcat('Results\latest\',CurrentDataset,'_Performance_Summary_DATA.mat');
%     save(filename,'ClassificationRate', 'ApproximationError');
%     
%     clear ClassificationRate ApproximationError;
    
    disp(['Dataset ' num2str(DatasetNo) ' Finished']);
    fprintf('\n');
    
end

    
 display('--------------------------------------------------------------------------------------------')
 display('Classification rate')
 display('  MLP_PSO    MLP_GWO     AVOA     MLP_HPSOGWO   MLP_GHPSOGWO   MLP_ADAMHPSOGWO   MLP_FODHPSOGWO')
  display(mean(ClassificationRate))
 display('--------------------------------------------------------------------------------------------')
 
 figure('Position',[500 500 660 290])
%Draw convergence curves
subplot(1,2,1);
hold on
title('Convergence Curves')
semilogy(mean(ConvergenceCurve1),'Color','r')
semilogy(mean(ConvergenceCurve2),'Color','k')
semilogy(mean(ConvergenceCurve3),'Color','b')
semilogy(mean(ConvergenceCurve4),'Color','y')
 semilogy(mean(ConvergenceCurve5),'Color','g')
 semilogy(mean(ConvergenceCurve6),'Color','c')
 semilogy(mean(ConvergenceCurve7),'Color','c')
semilogy(mean(ConvergenceCurve8),'Color','g')
xlabel('Generation');
ylabel('MSE');

axis tight
grid on
box on
legend('PSO','GWO','AVOA','HPSOGWO','5','6','7','JSOA')

%Draw classification rates
subplot(1,2,2);
hold on
title('Classification Accuracies')
bar(mean(ClassificationRate))
xlabel('Algorithm');
ylabel('Classification rate (%)');

grid on
box on
set(gca,'XTickLabel',{'PSO','GWO','AVOA','HPSOGWO','5','6','7','JSOA'});