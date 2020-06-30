clear;
clc;
load SVM 
% 训练数据和标签
% 数据有3个属性，4个类别
% 训练数据有30个
train_train = [train(1:4,:);train(5:11,:);train(12:19,:);train(20:30,:)];%手动划4分类
train_target = [target(1:4);target(5:11);target(12:19);target(20:30)];

% 测试数据和标签
% 测试数据有30个
test_simulation = [simulation(1:6,:);simulation(7:11,:);simulation(12:24,:);simulation(25:30,:)];
test_labels = [labels(1:6);labels(7:11);labels(12:24);labels(25:30)];

% train_train = normalization(train_train',1);
% test_simulation = normalization(test_simulation',1);
% train_train = train_train';
% test_simulation = test_simulation';


%           bestcv = 0;  
%           for log2c = -10:10,
%           for log2g = -10:10,
%             cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ',num2str(2^log2g)];%将训练集分为5类
%             cv = svmtrain(train_target, train_train, cmd);
%             if (cv >= bestcv),
%               bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%             end
%           end
%         end
%         fprintf('(best c=%g, g=%g, rate=%g)\n',bestc, bestg, bestcv);
%         cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
%         model = svmtrain(train_target, train_train, cmd);
 
model = svmtrain(train_target, train_train, '-c 2 -g 0.2 -t 1'); % 核函数为多项式核函数

[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_simulation, model);
predict_label

hold off
f=predict_label';
index1=find(f==1);
index2=find(f==2);
index3=find(f==3);
index4=find(f==4);
plot3(simulation(:,1),simulation(:,2),simulation(:,3),'o');
line(simulation(index1,1),simulation(index1,2),simulation(index1,3),'linestyle','none','marker','*','color','g');
line(simulation(index2,1),simulation(index2,2),simulation(index2,3),'linestyle','none','marker','<','color','r');
line(simulation(index3,1),simulation(index3,2),simulation(index3,3),'linestyle','none','marker','+','color','b');
line(simulation(index4,1),simulation(index4,2),simulation(index4,3),'linestyle','none','marker','>','color','y');
box;grid on;hold on;
xlabel('A');
ylabel('B');
zlabel('C');
title('支持向量机分析图');
