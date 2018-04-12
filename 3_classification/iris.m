clear, clc, close all
% Discriminate the 3 classes of flowers

% Load dataset
load iris_dataset.mat

% normalize data
x = zscore(irisInputs([1 2],:)');
t = irisTargets(1,:)'; % setosa vs all the rest

%gplotmatrix(x, [], t);

%CLASSIFICATION
% Perceptron (by hand)
% Check correct code on solutions (this did not work)
sigma = @(x, w)(1 ./ (1 + exp(-x * w))); % defined the activation function sigma

N = size(x, 1);
x_enh = [ones(N,1) x]; % adding the bias

w = rand(3, 1); % initialize randomly the weight vector
w_old = 2 * w;
alpha = 0.1;

while (norm(w-w_old) > 0.0001)
    w_old = w;
    for ii = 1:3
        grad(ii) = 1 / N * sum((sigma(x_enh, w) - t) * x_enh(:, ii));
    end
    w = w_old - alpha * grad';
end

%{
% Perceptron
net = perceptron;
net = train(net, x', t')
perc_t

% Logistic Regression
t = t + 1; % because mnrfit considers categorical classes corresponding to >0 int
[B, dev, stats] = mnrfit(x, t)

[t, ~] = find(irisTargets ~= 0)
[B_mul, dev_mul, stats_mul] = mnrfit(x, t)

% Naive Bayes
nb_model = fitcnb(x,t)
nb_model.DistributionParameters
nb_model.Prior


% KNN
knn_model = fitcknn(x, t, 'NumNeighbors', 3)
%}