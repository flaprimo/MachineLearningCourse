clear, clc, close all

n_points = 1000;
eps = 0.7;
func = @(x)(1 + 1 / 2 * x + 1 / 10 * x.^2);

%% Single model
x = 5 * rand(n_points,1);
t = func(x);
t_noisy = func(x) + eps * randn(n_points,1);
phi = [x x.^2];

lin_model = fitlm(x,t_noisy);
qua_model = fitlm(phi,t_noisy);

% Parameter space
real_par = [1 1/2 1/10];
best_lin_par = [7/12 1 0];

lin_c = [lin_model.Coefficients.Estimate' 0];
qua_c = qua_model.Coefficients.Estimate;

figure();
plot3(real_par(1),real_par(2),real_par(3),'bx');
hold on
grid on;
plot3(best_lin_par(1),best_lin_par(2),best_lin_par(3),'ro');
plot3(lin_c(1),lin_c(2),lin_c(3),'r+');
plot3(qua_c(1),qua_c(2),qua_c(3),'b+');
title('Parameter space');
xlabel('w_0');
ylabel('w_1');
zlabel('w_2');

% Input/output space

figure()
plot(x, t_noisy, 'r.')
hold on;
plot(x, predict(lin_model,x), 'g.');
plot(x, predict(qua_model,phi), 'b.');
title('Input/output space');
xlabel('x');
ylabel('y');


%% Multiple models
n_repetitions = 100;
n_points = 10000;
for ii = 1:n_repetitions
    % Samples generation
    x = 5 * rand(n_points,1);
    t = func(x);
    t_noisy = func(x) + eps * randn(n_points,1);
    phi = [x x.^2];

    lin_model = fitlm(x,t_noisy);
    qua_model = fitlm(phi,t_noisy);
    
    lin_coeff(ii,:) = [lin_model.Coefficients.Estimate' 0];
    qua_coeff(ii,:) = qua_model.Coefficients.Estimate;
end

figure();
plot3(real_par(1),real_par(2),real_par(3),'bx');
hold on
grid on;
plot3(best_lin_par(1),best_lin_par(2),best_lin_par(3),'ro');
plot3(lin_coeff(:,1),lin_coeff(:,2),lin_coeff(:,3),'r.');
plot3(qua_coeff(:,1),qua_coeff(:,2),qua_coeff(:,3),'b.');

title('Parameter space');
xlabel('w_0');
ylabel('w_1');
zlabel('w_2');

%% Bias and variance over a single point

x_new = 5 * rand();
t_new = func(x_new) + eps * randn(1,1);
x_enh_new = [1 x_new 0];
phi_new = [1 x_new x_new.^2];

for ii = 1:n_repetitions
    y_pred_lin(ii) = lin_coeff(ii,:) * x_enh_new';
    y_pred_qua(ii) = qua_coeff(ii,:) * phi_new';
end

error_lin = mean((t_new - y_pred_lin).^2);
bias_lin = mean(func(x_new) - y_pred_lin)^2;
variance_lin = var(y_pred_lin);
var_t_lin = error_lin - variance_lin - bias_lin;

error_qua = mean((t_new - y_pred_qua).^2);
bias_qua = mean(func(x_new) - y_pred_qua)^2;
variance_qua = var(y_pred_qua);
var_t_qua = error_qua- variance_qua - bias_qua;

disp('---Single point---');
disp(['Linear error: ' num2str(error_lin)]);
disp(['Linear bias: ' num2str(bias_lin)]);
disp(['Linear variance: ' num2str(variance_lin)]);
disp(['Linear sigma: ' num2str(var_t_lin) ' (unreliable)']);

disp(['Quadratic error: ' num2str(error_qua)]);
disp(['Quadratic bias: ' num2str(bias_qua)]);
disp(['Quadratic variance: ' num2str(variance_qua)]);
disp(['Quadratic sigma: ' num2str(var_t_qua) ' (unreliable)']);

%% Bias and variance over the x space

n_samples = 101;
x_new = linspace(0,5,n_samples)';
t_new = func(x_new) + eps * randn(n_samples,1);
x_enh_new = [ones(n_samples,1) x_new zeros(n_samples,1)];
phi_new = [ones(n_samples,1) x_new x_new.^2];

for ii = 1:n_repetitions
    y_pred_lin_all(ii,:) = lin_coeff(ii,:) * x_enh_new';
    y_pred_qua_all(ii,:) = qua_coeff(ii,:) * phi_new';
end

error_lin = sum(mean((repmat(t_new',n_repetitions,1) - ...
    y_pred_lin_all).^2)) / n_samples;
bias_lin = sum(mean(repmat(func(x_new'),n_repetitions,1) - ...
    y_pred_lin_all).^2) / n_samples;
variance_lin = sum(var(y_pred_lin_all)) / n_samples;
var_t_lin = (error_lin - bias_lin - variance_lin);

error_qua = sum(mean((repmat(t_new',n_repetitions,1) - ...
    y_pred_qua_all).^2)) / n_samples;
bias_qua = sum(mean(repmat(func(x_new'),n_repetitions,1) - ...
    y_pred_qua_all).^2) / n_samples;
variance_qua = sum(var(y_pred_qua_all)) / n_samples;
var_t_qua = (error_qua - bias_qua - variance_qua);

disp('---All dataset---')
disp(['Linear error: ' num2str(error_lin)]);
disp(['Linear bias: ' num2str(bias_lin)]);
disp(['Linear variance: ' num2str(variance_lin)]);
disp(['Linear sigma: ' num2str(var_t_lin)]);

disp(['Quadratic error: ' num2str(error_qua)]);
disp(['Quadratic bias: ' num2str(bias_qua)]);
disp(['Quadratic variance: ' num2str(variance_qua)]);
disp(['Quadratic sigma: ' num2str(var_t_qua)]);