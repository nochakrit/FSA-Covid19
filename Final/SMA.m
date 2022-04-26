% Source codes demo version 1.0
% ------------------------------------------------------------------------------------------------------------
% Main paper (Please refer to the main paper):
% Slime Mould Algorithm: A New Method for Stochastic Optimization
% Shimin Li, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, Seyedali Mirjalili
% Future Generation Computer Systems,2020
% DOI: https://doi.org/10.1016/j.future.2020.03.055
% https://www.sciencedirect.com/science/article/pii/S0167739X19320941
% ------------------------------------------------------------------------------------------------------------
% Website of SMA: http://www.alimirjalili.com/SMA.html
% You can find and run the SMA code online at http://www.alimirjalili.com/SMA.html

% You can find the SMA paper at https://doi.org/10.1016/j.future.2020.03.055
% Please follow the paper for related updates in researchgate: https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization
% ------------------------------------------------------------------------------------------------------------
%  Main idea: Shimin Li
%  Author and programmer: Shimin Li,Ali Asghar Heidari,Huiling Chen
%  e-Mail: simonlishimin@foxmail.com
% ------------------------------------------------------------------------------------------------------------
%  Co-author:
%             Huiling Chen(chenhuiling.jlu@gmail.com)
%             Mingjing Wang(wangmingjing.style@gmail.com)
%             Ali Asghar Heidari(aliasghar68@gmail.com, as_heidari@ut.ac.ir)
%             Seyedali Mirjalili(ali.mirjalili@gmail.com)
%
%             Researchgate: Ali Asghar Heidari https://www.researchgate.net/profile/Ali_Asghar_Heidari
%             Researchgate: Seyedali Mirjalili https://www.researchgate.net/profile/Seyedali_Mirjalili
%             Researchgate: Huiling Chen https://www.researchgate.net/profile/Huiling_Chen
% ------------------------------------------------------------------------------------------------------------
% _____________________________________________________
%  Co-author and Advisor: Seyedali Mirjalili
%
%         e-Mail: ali.mirjalili@gmail.com
%
%       Homepage: http://www.alimirjalili.com
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Max_iter: maximum iterations, N: populatoin size, Convergence_curve: Convergence curve
% To run SMA: [Destination_fitness,bestPositions,Convergence_curve]=SMA(N,Max_iter,lb,ub,dim,fobj)
function [Destination_fitness,bestPositions,Convergence_curve]=SMA(N,Max_iter,lb,ub,dim,fobj,mlpConfig)
disp('SMA is now tackling your problem');

% initialize position
bestPositions = zeros(1,dim);

% change this to -inf for maximization problems
Destination_fitness = inf;

% record the fitness of all slime mold
AllFitness = inf * ones(N,1);

% fitness weight of each slime mold
weight = ones(N,dim);

%Initialize the set of random solutions
X = initialization(N,dim,ub,lb);
Convergence_curve = zeros(1,Max_iter);

it = 1;                   %Number of iterations
lb = ones(1,dim).*lb;     % lower boundary
ub = ones(1,dim).*ub;     % upper boundary
z = 0.03;                 % parameter

% Main loop
while  it <= Max_iter
    
    %sort the fitness
    for i=1:N
        % Check if solutions go outside the search space and bring them back
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        AllFitness(i) = fobj(X(i,:),mlpConfig);
        %AllFitness(i) = test_functions(X(i,:),F_index,dim);
        %AllFitness(i) = cec17_func(X(i,:)',F_index);
    end
    
    [SmellOrder,SmellIndex] = sort(AllFitness);  %Eq.(2.6)
    worstFitness = SmellOrder(N);
    bestFitness = SmellOrder(1);
    
    S=bestFitness-worstFitness+eps;  % plus eps to avoid denominator zero
    
    %calculate the fitness weight of each slime mold
    for i=1:N
        for j=1:dim
            if i<=(N/2)  %Eq.(2.5)
                weight(SmellIndex(i),j) = 1+rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            else
                weight(SmellIndex(i),j) = 1-rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            end
        end
    end
    
    %update the best fitness value and best position
    if bestFitness < Destination_fitness
        bestPositions=X(SmellIndex(1),:);
        Destination_fitness = bestFitness;
    end
    
    a = atanh(-(it/Max_iter)+1);   %Eq.(2.4)
    b = 1-it/Max_iter;
    
    % Update the Position of search agents
    for i=1:N
        if rand<z     %Eq.(2.7)
            X(i,:) = (ub-lb)*rand+lb;
            %X(i,:) = Xnew(j);
        else
            p =tanh(abs(AllFitness(i)-Destination_fitness));  %Eq.(2.2)
            vb = unifrnd(-a,a,1,dim);  %Eq.(2.3)
            vc = unifrnd(-b,b,1,dim);
            
            for j=1:dim
                r = rand();
                A = randi([1,N]);  % two positions randomly selected from population
                B = randi([1,N]);
                if r<p    %Eq.(2.1)
                    X(i,j) = bestPositions(j)+ vb(j)*(weight(i,j)*X(A,j)-X(B,j));
                else
                    X(i,j) = vc(j)*X(i,j);
                end
            end
        end
    end
    Convergence_curve(it)=Destination_fitness;
    it=it+1;
    
    %ShowSurface(lb,ub,F_index); hold on;
    %
    %for i=1:N
    %    plot(X(i,1),X(i,2),'*');
    %end
    %
    %hold off;
    %drawnow;
    
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end

% _________________________________________________
% Gradient Search Rule
function GSR=GradientSearchRule(ro1,Best_X,Worst_X,X,Xr1,DM,eps,Xm,Flag)
nV = size(X,2);
Delta = 2.*rand.*abs(Xm-X);                            % Eq.(16.2)
Step = ((Best_X-Xr1)+Delta)/2;                         % Eq.(16.1)
DelX = rand(1,nV).*(abs(Step));                        % Eq.(16)

GSR = randn.*ro1.*(2*DelX.*X)./(Best_X-Worst_X+eps);   % Gradient search rule  Eq.(15)
if Flag == 1
    Xs = X - GSR + DM;                                   % Eq.(21)
else
    Xs = Best_X - GSR + DM;
end
yp = rand.*(0.5*(Xs+X)+rand.*DelX);                    % Eq.(22.6)
yq = rand.*(0.5*(Xs+X)-rand.*DelX);                    % Eq.(22.7)
GSR = randn.*ro1.*(2*DelX.*X)./(yp-yq+eps);            % Eq.(23)
end


function [z] = levy(n,m,beta)
% This function implements Levy's flight.
% For more information see
%'Multiobjective cuckoo search for design optimization Xin-She Yang, Suash Deb'.
% Coded by Hemanth Manjunatha on Nov 13 2015.
% Input parameters
% n     -> Number of steps
% m     -> Number of Dimensions
% beta  -> Power law index  % Note: 1 < beta < 2
% Output
% z     -> 'n' levy steps in 'm' dimension
num = gamma(1+beta)*sin(pi*beta/2); % used for Numerator

den = gamma((1+beta)/2)*beta*2^((beta-1)/2); % used for Denominator
sigma_u = (num/den)^(1/beta);% Standard deviation
u = random('Normal',0,sigma_u^2,n,m);

v = random('Normal',0,1,n,m);
z = u./(abs(v).^(1/beta));


end