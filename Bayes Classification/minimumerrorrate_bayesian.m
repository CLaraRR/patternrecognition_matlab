clear;
clc;
%% ��������dataset������ѵ�����ݺͲ�������,����shapeΪ[������������ά��] %%
load('./dataset.mat');
train_data = [A_train;B_train;C_train;D_train];
test_data = [A_test;B_test;C_test;D_test];
N1_train = size(A_train, 1); N2_train = size(B_train, 1); N3_train = size(C_train, 1); N4_train = size(D_train, 1); % ��������ѵ��������
N_train = N1_train + N2_train + N3_train + N4_train; % ѵ����������
N1_test = size(A_test, 1); N2_test = size(B_test, 1); N3_test = size(C_test, 1); N4_test = size(D_test, 1); % �������Ĳ�������������
N_test = N1_test + N2_test + N3_test + N4_test; % ������������
w = 4; % �����
n = 3; % ������

%% ��ʼ�������ݼ��� %%
% ��������ֵ
X1 = mean(A_train)'; X2 = mean(B_train)'; X3 = mean(C_train)'; X4 = mean(D_train)';
% ������Э�������
S1 = cov(A_train); S2 = cov(B_train); S3 = cov(C_train); S4 = cov(D_train); 
% ��Э�������������
S1_ = inv(S1); S2_ = inv(S2); S3_ = inv(S3); S4_ = inv(S4); 
% ��Э������������ʽ
S11 = det(S1); S22 = det(S2); S33 = det(S3); S44 = det(S4); 
% �������
Pw1 = N1_train/N_train; Pw2 = N2_train/N_train; Pw3 = N3_train/N_train; Pw4 = N4_train/N_train; 

%% ������������ĺ������ %%
for k = 1 : N_test
    P1 = -1/2*(test_data(k,:)'-X1)'*S1_*(test_data(k,:)'-X1)-1/2*log(S11)+log(Pw1);
    P2 = -1/2*(test_data(k,:)'-X2)'*S2_*(test_data(k,:)'-X2)-1/2*log(S22)+log(Pw2);
    P3 = -1/2*(test_data(k,:)'-X3)'*S3_*(test_data(k,:)'-X3)-1/2*log(S33)+log(Pw3);
    P4 = -1/2*(test_data(k,:)'-X4)'*S4_*(test_data(k,:)'-X4)-1/2*log(S44)+log(Pw4);
    P = [P1 P2 P3 P4];
    Pmax = max(P); % ȡ�������������һ��
    if Pmax == P1
        w = 1;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'ro');
        grid on;hold on;
    elseif Pmax == P2
        w = 2;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'b>');
        grid on;hold on;
    elseif Pmax == P3
        w = 3;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'g+');
        grid on;hold on;
    elseif Pmax == P4
        w = 4;
        plot3(test_data(k,1), test_data(k,2), test_data(k,3),'y*');
        grid on;hold on;
    else
        return
    end
end
    
