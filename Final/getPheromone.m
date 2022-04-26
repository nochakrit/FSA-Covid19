%  Function taken from BlackWidow Optimization Algorithm
%  available at:  https://doi.org/10.1155/2020/8856040
%  Programmer: Hernan Peraza    hperaza@ipn.mx
%*******************************************************
function [ o ] =  getPheromone(  fit, min, max )
    for i=1:size(fit,2)
         o(i)= (max-fit(i))/(max-min);
    end
end
 % Used in Jumping spider’ pheromone rates Eq.9
 %*******************************************************


