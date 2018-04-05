clear, clc, close all

load abalone_dataset.mat

idx = find(abaloneInputs(1,:) ~= 2);
dataset = abaloneInputs(:,idx);

x = zscore(dataset([2:8],:)');
t = dataset(1,:)';

%gplotmatrix(x, [], t);

% CLASSIFICATION
% Perceptron

net = perceptron;
net = train(net, x', t')