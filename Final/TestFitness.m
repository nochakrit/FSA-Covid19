
function [Acc, Err] = TestFitness(Function_name,Runno,mlpConfig,solution)

if Function_name=="F1"
    [xTrain, tTrain ,xTest, tTest] = DatasetInit("COVID_7d");
          
    gTest2 = zeros(size(tTrain)+size(tTest));
    gTest=gTest2(:,2);
    gTest(1:500)=tTrain;
    gPredict=NaN(size(gTest));
    gTest(501:end)=tTest;
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
   % Y=zeros(size(tTest));
    for agentNo = 1:Runno
               % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
     
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        %Performance of Testing
        %         [tmp,Index1] = max(Y,[],2);
        %         [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        tmpAcc(agentNo,1) = 0;
        tmpErr(agentNo,1) = mse((tTest) - Y);
        gPredict(501:end)=Y;
        %gPredict(501:end)=gPredict(501:end).*3553720;
        figure(2)
        semilogy(gTest.*3553720,'k')
        hold on
        semilogy(gPredict.*3553720,'r--')
        legend('Actual','Predictable')
        drawnow
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr)
    
end

if Function_name=="F2"
    [xTrain, tTrain ,xTest, tTest] = DatasetInit("COVID_14d");
          
    gTest2 = zeros(size(tTrain)+size(tTest));
    gTest=gTest2(:,1);
    gTest(1:500)=tTrain;
    gPredict=NaN(size(gTest));
    gTest(501:end)=tTest;
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
   % Y=zeros(size(tTest));
    for agentNo = 1:Runno
               % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
     
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        %Performance of Testing
        %         [tmp,Index1] = max(Y,[],2);
        %         [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        tmpAcc(agentNo,1) = 0;
        tmpErr(agentNo,1) = mse((tTest) - Y);
        gPredict(501:end)=Y;
        %gPredict(501:end)=gPredict(501:end).*3553720;
        figure(2)
        semilogy(gTest.*3553720,'k')
        hold on
        semilogy(gPredict.*3553720,'r--')
        legend('Actual','Predictable')
        drawnow
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr)
    
end

if Function_name=="F3"
    [xTrain, tTrain ,xTest, tTest] = DatasetInit("COVID_30d");
          
    gTest2 = zeros(size(tTrain)+size(tTest));
    gTest=gTest2(:,2);
    gTest(1:500)=tTrain;
    gPredict=NaN(size(gTest));
    gTest(501:end)=tTest;
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
   % Y=zeros(size(tTest));
    for agentNo = 1:Runno
               % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
     
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        %Performance of Testing
        %         [tmp,Index1] = max(Y,[],2);
        %         [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        tmpAcc(agentNo,1) = 0;
        tmpErr(agentNo,1) = mse((tTest) - Y);
        gPredict(501:end)=Y;
        %gPredict(501:end)=gPredict(501:end).*3553720;
        figure(2)
        semilogy(gTest.*3553720,'k')
        hold on
        semilogy(gPredict.*3553720,'r--')
        legend('Actual','Predictable')
        drawnow
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr)
    
end


if Function_name=="F4"
    [xTrain, tTrain ,xTest, tTest] = DatasetInit("COVID7_WINDOWX2");
          
    gTest2 = zeros(size(tTrain)+size(tTest));
    gTest=gTest2(:,2);
    gTest(1:500)=tTrain;
    gPredict=NaN(size(gTest));
    gTest(501:end)=tTest;
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
   % Y=zeros(size(tTest));
    for agentNo = 1:Runno
               % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
     
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        %Performance of Testing
        %         [tmp,Index1] = max(Y,[],2);
        %         [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        tmpAcc(agentNo,1) = 0;
        tmpErr(agentNo,1) = mse((tTest) - Y);
        gPredict(501:end)=Y;
        %gPredict(501:end)=gPredict(501:end).*3553720;
        figure(2)
        semilogy(gTest.*3553720,'k')
        hold on
        semilogy(gPredict.*3553720,'r--')
        legend('Actual','Predictable')
        drawnow
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr)
    
end

if Function_name=="F5"
    [xTrain, tTrain ,xTest, tTest] = DatasetInit("COVID7_WINDOWX3");
          
    gTest2 = zeros(size(tTrain)+size(tTest));
    gTest=gTest2(:,2);
    gTest(1:500)=tTrain;
    gPredict=NaN(size(gTest));
    gTest(501:end)=tTest;
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
   % Y=zeros(size(tTest));
    for agentNo = 1:Runno
               % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
     
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        %Performance of Testing
        %         [tmp,Index1] = max(Y,[],2);
        %         [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        tmpAcc(agentNo,1) = 0;
        tmpErr(agentNo,1) = mse((tTest) - Y);
        gPredict(501:end)=Y;
        %gPredict(501:end)=gPredict(501:end).*3553720;
        figure(2)
        semilogy(gTest.*3553720,'k')
        hold on
        semilogy(gPredict.*3553720,'r--')
        legend('Actual','Predictable')
        drawnow
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr)
    
end

if Function_name=="F6"
    [xTrain, tTrain ,xTest, tTest] = DatasetInit("COVID7_WINDOWX4");
          
    gTest2 = zeros(size(tTrain)+size(tTest));
    gTest=gTest2(:,2);
    gTest(1:500)=tTrain;
    gPredict=NaN(size(gTest));
    gTest(501:end)=tTest;
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
   % Y=zeros(size(tTest));
    for agentNo = 1:Runno
               % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
     
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        %Performance of Testing
        %         [tmp,Index1] = max(Y,[],2);
        %         [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        tmpAcc(agentNo,1) = 0;
        tmpErr(agentNo,1) = mse((tTest) - Y);
        gPredict(501:end)=Y;
        %gPredict(501:end)=gPredict(501:end).*3553720;
        figure(2)
        semilogy(gTest.*3553720,'k')
        hold on
        semilogy(gPredict.*3553720,'r--')
        legend('Actual','Predictable')
        drawnow
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr)
    
end

if Function_name=="F7"
    [xTrain, tTrain ,xTest, tTest] = DatasetInit("COVID7_WINDOWX5");
          
    gTest2 = zeros(size(tTrain)+size(tTest));
    gTest=gTest2(:,2);
    gTest(1:500)=tTrain;
    gPredict=NaN(size(gTest));
    gTest(501:end)=tTest;
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
   % Y=zeros(size(tTest));
    for agentNo = 1:Runno
               % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
     
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        %Performance of Testing
        %         [tmp,Index1] = max(Y,[],2);
        %         [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        tmpAcc(agentNo,1) = 0;
        tmpErr(agentNo,1) = mse((tTest) - Y);
        gPredict(501:end)=Y;
        %gPredict(501:end)=gPredict(501:end).*3553720;
        figure(2)
        semilogy(gTest.*3553720,'k')
        hold on
        semilogy(gPredict.*3553720,'r--')
        legend('Actual','Predictable')
        drawnow
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr)
    
end




end