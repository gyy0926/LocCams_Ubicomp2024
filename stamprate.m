function [m] = stamprate(timestamp)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
s = 100;
L = floor(length(timestamp)/s);
rate = zeros(1,L);
for i = 1:L
    if i == 1
        rate(i) = s/timestamp(s);
    else
        rate(i) = s/(timestamp(s*i)-timestamp(s*i-s));
    end
end

step = floor(L/4);
s1 = sort(rate(1:step));
m1 = mean(s1(1:end-1));
s2 = sort(rate(1+step:step*2));
m2 = mean(s2(1:end-1));
s3 = sort(rate(1+step*2:step*3));
m3 = mean(s3(1:end-1));
s4 = sort(rate(1+step*3:end));
m4 = mean(s4(1:end-1));
m5 = (s1(end)+s2(end)+s3(end)+s4(end))/4;
m6 = (m1+m2+m3+m4)/4;
m = (m5-m6)/m6;
end
