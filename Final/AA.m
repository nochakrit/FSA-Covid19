clear all;
clc;

out = load('ccvv.txt');
%dif=zeros(size(out);
growth = zeros(size(out));

for i=1:size(growth)-1
    growth(i)=((out(i+1)-out(i))./out(i))*100;
end
