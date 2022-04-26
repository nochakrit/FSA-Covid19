%Visualization
tiledlayout(2,2)

selectedExperiment = '27052564';

for currentDataset = 1:size(DatasetName,1)
    
    filename = strcat('Results\',selectedExperiment,'\',string(DatasetName(currentDataset)),'_Performance_Summary_DATA.mat');
    load(filename,'ClassificationRate');
    load(filename,'ApproximationError');
    
    figure(currentDataset)
    nexttile
    plot(ClassificationRate(1,:),'-')
    hold on
    plot(ClassificationRate(2,:),'-')
    plot(ClassificationRate(3,:),'-')
    plot(ClassificationRate(4,:),'-')
    plot(ClassificationRate(5,:),'-')
    plot(ClassificationRate(6,:),'-*')
    plot(ClassificationRate(7,:),'-')
    plot(ClassificationRate(8,:),'-')
    plot(ClassificationRate(9,:),'-')
    hold off
    title('Testing Accuracy: ' + string(DatasetName(currentDataset)));
    xlabel('No. of Hidden Node');
    ylabel('Classification Rate');
    legend('PSO','GWO','IGWO','HPSOGWO','GHPSOGWO','ADAMHPSOGWO','SMA','GBO','WOA');
    
    nexttile
    plot(ApproximationError(1,:),'-')
    hold on
    plot(ApproximationError(2,:),'-')
    plot(ApproximationError(3,:),'-')
    plot(ApproximationError(4,:),'-')
    plot(ApproximationError(5,:),'-')
    plot(ApproximationError(6,:),'-*')
    plot(ApproximationError(7,:),'-')
    plot(ApproximationError(8,:),'-')
    plot(ApproximationError(9,:),'-')
    hold off
    title('Testing Error: ' + string(DatasetName(currentDataset)));
    xlabel('No. of Hidden Node');
    ylabel('MSE');
    legend('PSO','GWO','IGWO','HPSOGWO','GHPSOGWO','ADAMHPSOGWO','SMA','GBO','WOA');
    drawnow;
    
    clear ClassificationRate ApproximationError;
    
end