% Grey Wolf Optimizer
function [Alpha_score,Alpha_pos,Convergence_curve]=HPSOGWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,mlpConfig)
disp('HPSOGWO is now tackling your problem');

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
            velocity(i,j)=w*(velocity(i,j)+C1*r1*(X1-Positions(i,j))+C2*r2*(X2-Positions(i,j))+C3*r3*(X3-Positions(i,j)));
            
            % positions update
            Positions(i,j)=Positions(i,j)+velocity(i,j);
        end
    end
    
    l=l+1;
    Convergence_curve(l)=Alpha_score;
end
end

function xhold=linnikrndX(alpha,scale,m, n)
xhold = laprnd(m, n, 0, 1);
%xhold = randl(m,n); %Laplacian distributed pseudorandom numbers.
SE = sign(rand(m,n)-0.5) .* xhold;
%U = rand(1,n);
U = rand(m,n);
xhold = (sin(0.5*pi*alpha).*tan(0.5*pi*(1-alpha*U))-cos(0.5*pi*alpha)).^(1/alpha);
xhold = scale * SE ./ xhold;
end

function y  = laprnd(m, n, mu, sigma)
%LAPRND generate i.i.d. laplacian random number drawn from laplacian distribution
%   with mean mu and standard deviation sigma.
%   mu      : mean
%   sigma   : standard deviation
%   [m, n]  : the dimension of y.
%   Default mu = 0, sigma = 1.
%   For more information, refer to
%   http://en.wikipedia.org./wiki/Laplace_distribution

%   Author  : Elvis Chen (bee33@sjtu.edu.cn)
%   Date    : 01/19/07

%Check inputs
if nargin < 2
    error('At least two inputs are required');
end

if nargin == 2
    mu = 0; sigma = 1;
end

if nargin == 3
    sigma = 1;
end

% Generate Laplacian noise
u = rand(m, n)-0.5;
b = sigma / sqrt(2);
y = mu - b * sign(u).* log(1- 2* abs(u));
end



