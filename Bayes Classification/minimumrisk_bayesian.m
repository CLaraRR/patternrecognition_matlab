clear;
clc;
%% 加载样本dataset，包含训练数据和测试数据,数据shape为[样本数，特征维数] %%
load('./dataset.mat');
train_data = [A_train;B_train;C_train;D_train];
test_data = [A_test;B_test;C_test;D_test];
N1_train = size(A_train, 1); N2_train = size(B_train, 1); N3_train = size(C_train, 1); N4_train = size(D_train, 1); % 各个类别的训练样本数
N_train = N1_train + N2_train + N3_train + N4_train; % 训练样本总数
N1_test = size(A_test, 1); N2_test = size(B_test, 1); N3_test = size(C_test, 1); N4_test = size(D_test, 1); % 各个类别的测试样本的数量
N_test = N1_test + N2_test + N3_test + N4_test; % 测试样本总数
w = 4; % 类别数
n = 3; % 特征数

%% 初始样本数据计算 %%
% 求样本均值
X1 = mean(A_train)'; X2 = mean(B_train)'; X3 = mean(C_train)'; X4 = mean(D_train)';
% 求样本协方差矩阵
S1 = cov(A_train); S2 = cov(B_train); S3 = cov(C_train); S4 = cov(D_train); 
% 求协方差矩阵的逆矩阵
S1_ = inv(S1); S2_ = inv(S2); S3_ = inv(S3); S4_ = inv(S4); 
% 求协方差矩阵的行列式
S11 = det(S1); S22 = det(S2); S33 = det(S3); S44 = det(S4); 
% 先验概率
Pw1 = N1_train/N_train; Pw2 = N2_train/N_train; Pw3 = N3_train/N_train; Pw4 = N4_train/N_train; 
%% 定义损失函数 %%
loss = ones(4) - diag(diag(ones(4))); % 0-1损失函数；一个4*4矩阵，除了对角线元素为0，其他全为1
%% 计算测试样本的后验概率 %%
for k = 1 : N_test
    P1 = -1/2*(test_data(k,:)'-X1)'*S1_*(test_data(k,:)'-X1)-1/2*log(S11)+log(Pw1);
    P2 = -1/2*(test_data(k,:)'-X2)'*S2_*(test_data(k,:)'-X2)-1/2*log(S22)+log(Pw2);
    P3 = -1/2*(test_data(k,:)'-X3)'*S3_*(test_data(k,:)'-X3)-1/2*log(S33)+log(Pw3);
    P4 = -1/2*(test_data(k,:)'-X4)'*S4_*(test_data(k,:)'-X4)-1/2*log(S44)+log(Pw4);
    % 计算分别采取不同决策所带来的风险
    risk1 = loss(1,1)*P1 + loss(1,2)*P2 + loss(1,3)*P3 + loss(1,4)*P4;
    risk2 = loss(2,1)*P1 + loss(2,2)*P2 + loss(2,3)*P3 + loss(2,4)*P4;
    risk3 = loss(3,1)*P1 + loss(3,2)*P2 + loss(3,3)*P3 + loss(3,4)*P4;
    risk4 = loss(4,1)*P1 + loss(4,2)*P2 + loss(4,3)*P3 + loss(4,4)*P4;
    risk = [risk1 risk2 risk3 risk4];
    min_risk = min(risk); % 找出最小风险值
    if min_risk == risk1
        w = 1;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'ro');
        grid on;hold on;
    elseif min_risk == risk2
        w = 2;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'b>');
        grid on;hold on;
    elseif min_risk == risk3
        w = 3;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'g+');
        grid on;hold on;
    elseif min_risk == risk4
        w = 4;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'y*');
        grid on;hold on;
    else
        return
    end
end
    