function [point,corrs] = ExtractSeq(csi_ampt,inp_flag)
%UNTITLED3 此处显示有关此函数的摘要
%   point表示子序列分割点
%   corrs表示子序列合集
%   csi_ampt表示CSI的幅值矩阵
%   inp_flag表示传入的序列包含三个LOS还是两个LOS，用于打标签

an0 = csi_ampt;
data_corr = zeros(10,size(an0,1));
corrs = [];
for k = 1:10
    for i=1:size(an0,1)-k
        data_corr(k,i) = min(corrcoef(an0(i,:),an0(i+k,:)),[],'all');  %测量值之间的相关性
    end
    data_corr(k,size(an0,1)) = mean(data_corr(k,1:size(an0,1)-k));
end

var_flag = 1;
flag=0;
point = zeros(8,1);

for i =1:size(an0,1)
    if flag == 0
        cha = data_corr(1,i);
        if cha>0.9
            point(var_flag) = i;
            var_flag = var_flag+1;
            flag = 1;
            if var_flag>5
                if point(var_flag-3)-point(var_flag-4)<20
                   point(var_flag-4) = point(var_flag-2);
                   point(var_flag-3) = i;
                   var_flag = var_flag-2;
                end    
            end
        else   
            continue;
        end
    else
        cha = data_corr(1,i);
        cha_i = i-point(var_flag-1);
        if cha<0.9 && cha_i>100
            point(var_flag) = i-1;
            var_flag = var_flag+1;
            flag = 0;
        elseif cha<0.9 && cha_i<=100
            var_flag = var_flag-1;
            flag = 0;
        else
            continue;
        end
    end
end
if point(8) == 0
    point(8) = 2000;
end
point = point';

if point(2)-point(1)<=100 || point(4)-point(3)<=100 || point(6)-point(5)<=100 || point(8)-point(7)<=100 || point(1)>300 || point(8)<1000
    return;
end
max_s1 = zeros(1,point(2)-point(1)-98);
for i = 1:(point(2)-point(1)-98)
    max_s1(i) = sum(data_corr(1,point(1)+i-1:point(1)+i+98));
end
[~,I] = max(max_s1);
s1 = csi_ampt(point(1)+I-1:point(1)+I+98,:);
max_s2 = zeros(1,point(4)-point(3)-98);
for i = 1:(point(4)-point(3)-98)
    max_s2(i) = sum(data_corr(1,point(3)+i-1:point(3)+i+98));
end
[~,I] = max(max_s2);
s2 = csi_ampt(point(3)+I-1:point(3)+I+98,:);
max_s3 = zeros(1,point(6)-point(5)-98);
for i = 1:(point(6)-point(5)-98)
    max_s3(i) = sum(data_corr(1,point(5)+i-1:point(5)+i+98));
end
[~,I] = max(max_s3);
s3 = csi_ampt(point(5)+I-1:point(5)+I+98,:);
max_s4 = zeros(1,point(8)-point(7)-98);
for i = 1:(point(8)-point(7)-98)
    max_s4(i) = sum(data_corr(1,point(7)+i-1:point(7)+i+98));
end
[~,I] = max(max_s4);
s4 = csi_ampt(point(7)+I-1:point(7)+I+98,:);

%% 1代表LOS，0代表NLOS
temp1 = zeros(400,1);
temp1(1:200) = 1;
temp0 = zeros(400,1);
temp0(1:300) = 1;

if inp_flag == 0
    corrs1 = [s1',s4',s2',s3']';
    corrs = [corrs1,temp0];  
else
    corrs1 = [s1',s4',s3',s2']';
    corrs = [corrs1,temp1];    
end

end

