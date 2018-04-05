clear, clc, close all
% Predict petal width (t_n) of Iris setosa using petal length (x_n)
% * hp space: w_0 + x_n * w_1
% * loss: RSS(w)
% * optimization: LS method

% Load dataset
load iris_dataset.mat

% DATA EXPLORATION: plot data for preliminary analysis
% we can see that there's a linear relation in box (3;4) and consequently
% on (4;3)
%figure(), gplotmatrix(irisInputs');
x = irisInputs(3,:)'; % Petal length
t = irisInputs(4,:)'; % Petal width

% PRE-PROCESSING
% normalization
x = zscore(x);
t = zscore(t);
%figure(), plot(x, t, 'bo');

% FITTING
% model we want to fit
fit_specifications = fittype(...
    {'1', 'x'}, ... % model type
    'independent', 'x', ...
    'dependent', 't', ...
    'coefficients', {'w0', 'w1'}) % coefficients
% fit results and gof parameters that let us understand how good is the
% fitted model
[fitresults, gof] = fit(x, t, fit_specifications)

% alternative
% more general, allows us to do more regressions
%ls_model = fitlm(x, t)

% OPTIMIZATION
n_sample = length(x);
Phi = [ones(n_sample, 1) x x.^2]
lambda = 10^(-10);
ridge_coeff = ridge(t, Phi, lambda)
[lasso_coeff, lasso_fit] = lasso(Phi, t);