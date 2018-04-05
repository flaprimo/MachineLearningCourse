clear, clc, close all

% http://blog.minitab.com/blog/adventures-in-statistics-2/how-to-interpret-regression-analysis-results-p-values-and-coefficients

% Load data
load carsmall.mat
X = [Acceleration, Weight, Displacement, Horsepower];

% Normalize data
X = (X - repmat(nanmean(X),100,1)) ./ repmat(nanstd(X),100,1);

% Fit data wrt Acceleration
fitted_model = fitlm(X(:,2:end),X(:,1))
betahat = fitted_model.Coefficients.Estimate % get the parameters