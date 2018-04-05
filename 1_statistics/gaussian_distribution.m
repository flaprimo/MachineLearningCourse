%create distribution
X = makedist('Normal');

% probability density function
x = -3:.1:3; % realizations
pdf_normal = pdf(X, x);

X.pdf(5) %pdf at x = 5
X.cdf(3) %CDF for x = 3
X.icdf(0.05) %inverse CDF for alpha = 0.05


plot(x, pdf_normal, 'LineWidth', 2)