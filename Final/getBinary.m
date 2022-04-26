%  Function taken from BlackWidow Optimization Algorithm
%  available at:  https://doi.org/10.1155/2020/8856040
%  Programmer: Hernan Peraza    hperaza@ipn.mx
%*******************************************************
function [ val] = getBinary( )
if rand() < 0.5
     val= 0;
else
     val=1;
end
 % Used in Jumping spider’ pheromone rates Eq.9
%*******************************************************
