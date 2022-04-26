% Flamingo Search Algorithm: A New Swarm Intelligence Optimization Algorithm

% CITE:
% W. Zhiheng and L. Jianhua, "Flamingo Search Algorithm: A New Swarm Intelligence Optimization Algorithm,"
% in IEEE Access, vol. 9, pp. 88564-885/ๆ ะุ82, 2021, doi: 10.1109/ACCESS.2021.3090512.

function [best_cost,best_x,convergence_curve] = FSA(search_agent_no,max_nfe,lb,ub,dim,fobj,mlpConfig)
%disp('FSA is now tackling your problem');

rng(sum(100*clock));

% ------ OUTPUT ------ %
best_cost = inf;
best_x = zeros(1,dim);
% convergence_curve = zeros(1,max_nfe/search_agent_no);
% ------ OUTPUT ------ %

% ===== PARAMETERS ===== %
MPb = 0.1; % proportion of migrating flamingos
% ===== PARAMETERS ===== %


% Initialize the population
X = initialization(search_agent_no,dim,ub,lb);

% Initialize fitness value
fitness = zeros(search_agent_no,1);
for i = 1:search_agent_no
    fitness(i,1) = feval(fobj,X(i,:),mlpConfig);
end

% Rank the fitness values (best to worst : ascending)
[X, Cost, ~] = PopSort(X,fitness);

% Find the current best
current_best(1,:) = X(1,:);
current_best_fitness = Cost(1,:);

nfe = 0;
convergence_curve_step = 1;

while nfe < max_nfe
   
    R = rand();
    MPr = round(R * search_agent_no * (1 - MPb));
    MP0 = MPb;
    MPt = search_agent_no - round(MP0 * search_agent_no) - MPr;
    
    for i = 1:round(MPb * search_agent_no)
        for j = 1:dim
            gassiun_random_number = normrnd(0,1.2);
            X(i,j) = X(i,j) + gassiun_random_number * (X(1,j) - X(i,j)); % eq. 3
        end
    end
    
    for i = 1 + round(MP0 * search_agent_no):round(MP0 * search_agent_no) + MPr
        for j = 1:dim
            while true
                r1 = randperm(3,1) - 2;
                if r1~=0,break,end
            end
            while true
                r2 = randperm(3,1) - 2;
                if r2~=0,break,end
            end
            
            G1 = normrnd(0,1);
            G2 = normrnd(0,1);
            K = chi2rnd(8);
            
            X(i,j) = (X(i,j) + r1 * X(1,j) + G2 * abs(G1 * X(1,j) + r2 * X(i,j))) / K; % eq. 2
        end
    end
    
    for i = round(MP0 * search_agent_no) + MPr + 1:search_agent_no
        for j = 1:dim
            gassiun_random_number = normrnd(0,1.2);
            X(i,j) = X(i,j) + gassiun_random_number * (X(1,j) - X(i,j)); % eq. 3
        end
    end
    
    % Boundary detection
    for i = 1:search_agent_no
        for j = 1:dim
            X(i,j) = max(X(i,j),lb());
            X(i,j) = min(X(i,j),ub());
        end
    end
    
    % Calculate tje fitness value
    for i = 1:search_agent_no
        fitness(i,1) = feval(fobj,X(i,:),mlpConfig);
        nfe = nfe + 1; % nfe update
    end
    
    % Rank the fitness values (best to worst : ascending)
    [X, Cost, ~] = PopSort(X,fitness);
    
    % Find the current best
    current_best(1,:) = X(1,:);
    current_best_fitness = Cost(1,:);
    
    best_x = current_best(1,:);
    best_cost = current_best_fitness(1,:);
    convergence_curve(convergence_curve_step) = current_best_fitness(1,:);
    
    convergence_curve_step = convergence_curve_step + 1; % Convergence step update
    iteration = round(nfe/search_agent_no);
    best_cost;
    iteration_run=[iteration,best_cost]
   
    
    
end
end
%-----------------------------------------------------------------------------------------------------
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

function [sorted_population, sorted_fitness, sorted_index] = PopSort(input_pop,input_fitness)

% Sort the population members from best to worst
popsize = size(input_pop,1);
Cost = zeros(popsize,1);
sorted_index = zeros(popsize,1);

for i = 1 : popsize
    Cost(i,1) = input_fitness(i,1);
end

[Cost, sorted_index] = sort(Cost,'ascend');
position = zeros(popsize, size(input_pop,2));

for i = 1 : popsize
    position(i,:) = input_pop(sorted_index(i,1),:);
end

for i = 1 : popsize
    sorted_population(i,:) = position(i,:);
    sorted_fitness(i,1) = Cost(i,1);
end
end