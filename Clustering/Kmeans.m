clear all;
clc;
%% 加载样本dataset，包含训练数据和测试数据,数据shape为[样本数，特征维数] %%
% 样本一共有4类
load('dataset.mat');
data = [A_test;B_test;C_test;D_test];
 [IDX,C,SUMD,D] = kmeans(data,4); % K=4
 % IDX:聚类结果
 % C：聚类中心
 % SUMD：每一个样本到该聚类中心的距离和
 % D：每一个样本到各个聚类中心的距离
 
plot3(data(:,1),data(:,2),data(:,3),'*');
grid;
D = D';
minD=min(D);
index1 = find(D(1,:) ==min(D))
index2 = find(D(2,:) ==min(D))
index3 = find(D(3,:) ==min(D))
index4 = find(D(4,:) ==min(D))
line(data(index1,1),data(index1,2),data(index1,3),'linestyle', 'none','marker','*','color','g');
line(data(index2,1),data(index2,2),data(index2,3),'linestyle', 'none','marker','*','color','r');
line(data(index3,1),data(index3,2),data(index3,3),'linestyle', 'none','marker','+','color','b');
line(data(index4,1),data(index4,2),data(index4,3),'linestyle', 'none','marker','+','color','y');
title('Kmeans聚类图');
xlabel('第一特征坐标');
ylabel('第二特征坐标');
zlabel('第三特征坐标');



