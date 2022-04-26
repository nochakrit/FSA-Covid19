function [weightHidden, biasHidden, weightOutput, biasOutput] = MLPWeightInit(solution,dimensionOfdata,sizeOfhidden,sizeOfoutput)

% This function for MLP optimized weight assignation

solution_position = 1; % Solution from optimizer

for i = 1:dimensionOfdata %size(xTrain,2)
    for j = 1:sizeOfhidden %L
        weightHidden(i,j) = solution(1,solution_position); %Weight of Total Hidden Node
        solution_position = solution_position + 1; % Move next
    end
end

for j = 1:sizeOfhidden %L
    biasHidden(1,j) = solution(1,solution_position); %Weight of Bias Hidden Node
    solution_position = solution_position + 1; % Move next
end

for i = 1:sizeOfhidden %L
    for j = 1:sizeOfoutput %O
        weightOutput(i,j) = solution(1,solution_position); %Weight of Output Node
        solution_position = solution_position + 1; % Move next
    end
end

for j = 1:sizeOfoutput %O
    biasOutput(1,j) = solution(1,solution_position); %Weight of Bias Output Node
    solution_position = solution_position + 1; % Move next
end

end