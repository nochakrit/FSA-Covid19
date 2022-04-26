% Copyright 2021 by Hernan Peraza at IPN CICATA Altamira.                  %
% All rights reserved. This source code or any portion thereof may not be  %
% reproduced or used in any manner whatsoever without the express written  %
% permission of the author. For scientific purposes include the publication
% reference.                                               %
% Public Registry of Copyright number: 03-2020-012211265800-01
%______________________________________________________________________%
%  Jumping Spider Optimization Algorithm (JSOA) source codes demo 1.0     %
%  Developed in MATLAB R2018a(9.2)                                        %
%  Authors:  Hernan Peraza  -  Fernando Ruiz                              %
%  e-Mail: hperaza@ipn.mx                                                 %
%   Main paper:                                                           %
%_________________________________________________________________________%
function [vMin,theBestVct,Convergence_curve]=JSOA(pop_size,max_iter,lower_bound,upper_bound,variables_no,fobj,mlpConfig)

parameters.SearchAgents = pop_size;
parameters.dim =variables_no;
parameters.ub = upper_bound;
parameters.lb = lower_bound;
parameters.maxIteration = max_iter;
parameters.fobj = fobj;

format long;
Positions=initialization(parameters.SearchAgents,parameters.dim,parameters.ub,parameters.lb);
 for i=1:size(Positions,1)
      Fitness(i)=parameters.fobj(Positions(i,:),mlpConfig);  % get fitness     
 end
[vMin minIdx]= min(Fitness);           
[vMax maxIdx]= max(Fitness); 
theBestVct=  Positions(minIdx,:);       % the best vector 
theWorstVct= Positions(maxIdx,:);       % the worst vector
% Convergence_curve=zeros(1,parameters.maxIteration);
% Convergence_curve(1)= vMin;
pheromone= getPheromone( Fitness, vMin, vMax);  % Used in Jumping spider’ pheromone rates Eq.9
gravity= 9.80665; %m/seg^2  
vo= 100; % 100 mm/seg
for t=1:parameters.maxIteration       
   for r=1:parameters.SearchAgents
       if rand() < 0.5  % Attack, persecution and jumping on the prey 
               if rand()  < 0.5
                     %****************************************************************
                     % Jumping on the prey represented by equation of
                     % projectile motion. Eq. 6 on paper 
                     radians=  90* rand()*pi/180; 
                     v(r,:)=  (Positions(r,:) .* tan(radians))-((gravity .*(Positions(r,:).^2)) ./ (2*(vo^2)*(cos(radians)^2)));
                     %***************************************
               else
                    % Persecution represented by the uniformly accelerated
                    % rectilinear motion. Eq.2 on paper
                    ban=1;
                     while(ban)
                            r1= round(1+ (parameters.SearchAgents-1)* rand());
                            if (r ~= r1)
                                 ban=0;
                            end 
                     end
                     v(r,:)= 0.5 * (Positions(r,:) -Positions(r1,:)); 
                 end
       else   % Searching for prey 
             if rand < 0.5    % Global search. Eq. 8 on paper
                  e1=  CauchyRand(0,1);
                  v(r,:)=  theBestVct +( theBestVct-theWorstVct)  * e1;                
             else             % Local search. Eq. 7 on paper 
                 walk= -2 + 4 * rand(); % -2 < d < 2   Uniformly distributed pseudorandom numbers
                 e=  randn(); % Normally distributed pseudorandom numbers.
                  v(r,:)= theBestVct + walk*(0.5-e);
             end
       end       
       if pheromone(r) <= 0.3  % Jumping spider’ pheromone rates. Eq.10, Algorithm 1 on paper.
             band=1;
             while band
               r1= round(1+ (parameters.SearchAgents-1)* rand());
               r2= round(1+ (parameters.SearchAgents-1)* rand());
               if r1 ~= r2
                   band=0;
               end
             end  % Eq.10 on paper
                  v(r,:)=   theBestVct + (Positions(r1,:)-((-1)^getBinary)*Positions(r2,:))/2;
       end    
     %****************************************************************
        Flag4ub=v(r,:)>parameters.ub;
        Flag4lb=v(r,:)<parameters.lb;
        v(r,:)=(v(r,:).*(~(Flag4ub+Flag4lb)))+parameters.ub.*Flag4ub+parameters.lb.*Flag4lb;
    % Evaluate new solutions
     Fnew= parameters.fobj(v(r,:),mlpConfig);
     % Update if the solution improves
     if Fnew <= Fitness(r)
        Positions(r,:)= v(r,:);
        Fitness(r)= Fnew;
     end
     if Fnew <= vMin
         theBestVct= v(r,:);   
         vMin= Fnew;           
     end 
   end
   [vMax maxIdx]= max(Fitness); 
   theWorstVct= Positions(maxIdx,:);
   % update max and pheromons
   [vMax maxIdx]= max(Fitness);
   pheromone= getPheromone( Fitness, vMin, vMax);
   Convergence_curve(t)= vMin; 
  end
end
%***********************************[End JSOA Algorithm]
