% Grey Wolf Optimizer
function [Alpha_score,Alpha_pos,Convergence_curve]=FODHPSOGWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,mlpConfig)
disp('FODHPSOGWO is now tackling your problem');

%% Fractional Order Darwinian PSO parameters Part I %%

% fodpso(func,xmin,xmax,type,population,nswarms,iterations,alfa)

% dimension of the problem
%N_PAR=length(argnames(fun));
%N_PAR=dim;

% xmin - minimum value of xi. size(xmin,2)=number of xi variables. Default -100.
%xmin=-100*ones(1,N_PAR);

% xmax - maximum value of xi. size(xmax,2)=number of xi variables. Default 100.
%xmax=100*ones(1,N_PAR);

% type - minimization 'min' or maximization 'max' of the problem. Default 'min'.
%type='min';

% population - number of particles within each swarm. Default 20.
%population=20;

% nswarms - number of starting swarms. Default 5.
%nswarms=5;

% iterations - number of iterations. Default 500.
%iterations=500;

%% Fractional Order Darwinian PSO parameters Part II %%

% alfa - fractional coefficient. Default 0.6.
alfa=0.6;
PHI1 = 1.5;  %default a 1.5
PHI2 = 1.5;  %default a 1.5
PHI3 = 1.5;  %default a 1.5
PHI4 = 1.5;  %default a 1.5
W = 0.9;     %default a 0.9

%%

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);
Convergence_curve=zeros(1,Max_iter);
velocity = .3*randn(SearchAgents_no,dim) ;
%w=0.5+rand()/2;
l=0;% Loop counter

% Main loop
while l<Max_iter
    
    w=0.5+rand()/2; %ksn
    
    for i=1:size(Positions,1)
        
        % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate objective function for each search agent
        fitness=fobj(Positions(i,:),mlpConfig);
        
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end
    end
    
    
    a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            %C1=2*r2; % Equation (3.4)
            C1=0.5;
            
            D_alpha=abs(C1*Alpha_pos(j)-w*Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
            
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            %C2=2*r2; % Equation (3.4)
            C2=0.5;
            
            D_beta=abs(C2*Beta_pos(j)-w*Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2
            
            r1=rand();
            r2=rand();
            r3=rand();
            
            A3=2*a*r1-a; % Equation (3.3)
            %C3=2*r2; % Equation (3.4)
            C3=0.5;
            
            D_delta=abs(C3*Delta_pos(j)-w*Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3
            
            % velocity updation
            %velocity(i,j)=w*(velocity(i,j)+C1*r1*(X1-Positions(i,j))+C2*r2*(X2-Positions(i,j))+C3*r3*(X3-Positions(i,j)));
            
            gaux = 1;
            %i=1;
            
            randnum1_alpha=rand();
            randnum1_beta=rand();
            randnum1_delta=rand();
            randnum2=rand();
            
            %vbef3=0;
            vbef2=0;
            vbef1=0;
            
            vbef3=vbef2;
            vbef2=vbef1;
            vbef1=velocity(i,j);
            
            % FODPSO + GWO
            velocity(i,j) = W * (alfa*velocity(i,j) + (1/2)*alfa*vbef1 + (1/6)*alfa*(1-alfa)*vbef2 + (1/24)*alfa*(1-alfa)*(2-alfa)*vbef3) + randnum1_alpha*(PHI1*(X1-Positions(i,j))) ...
                + randnum1_beta*(PHI2*(X2-Positions(i,j))) + randnum1_delta*(PHI3*(X3-Positions(i,j))) + randnum2.*(PHI4.*(gaux*Alpha_score-Positions(i,j)));
            
            % Fractional-Order DPSO
            %v = W*(alfa*v + (1/2)*alfa*vbef1 + (1/6)*alfa*(1-alfa)*vbef2 + (1/24)*alfa*(1-alfa)*(2-alfa)*vbef3) + randnum1.*(PHI1.*(xBest-x)) + randnum2.*(PHI2.*(gaux*gBest-x));
            
            N=SearchAgents_no;            
            vmin=-(max(ub)-min(lb))/(N*5);
            vmax=(max(ub)-min(lb))/(N*5);
            
            velocity(i,j) = ( (velocity(i,j) <= vmin)*vmin ) + ( (velocity(i,j) > vmin)*velocity(i,j) );
            velocity(i,j) = ( (velocity(i,j) >= vmax)*vmax ) + ( (velocity(i,j) < vmax)*velocity(i,j) );
            
            % positions update
            Positions(i,j)=Positions(i,j)+velocity(i,j);
        end
    end
    
    l=l+1;
    Convergence_curve(l)=Alpha_score;
end
end



