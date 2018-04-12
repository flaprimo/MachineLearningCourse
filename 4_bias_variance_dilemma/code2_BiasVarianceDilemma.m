clear
clc
close all

n_tot = 40;

eps = 1.5;
func = @(x)((0.5 - x) .* (5 - x) .* (x - 3));

x = 5 * rand(n_tot,1);
y = func(x) + eps * randn(n_tot,1);

%shuffling is not needed here
n_trai = 20;
n_vali = 10;
n_test = 10;

x_trai = x(1:n_trai);
y_trai = y(1:n_trai);

x_vali = x(n_trai+1:n_trai+n_vali);
y_vali = y(n_trai+1:n_trai+n_vali);

x_test = x(n_trai+n_vali+1:end);
y_test = y(n_trai+n_vali+1:end);

figure();
plot(x_trai, y_trai, 'x');
hold on;
plot(x_test, y_test, 'x');
plot(x_vali, y_vali, 'x');
plot(linspace(0,5,100), func(linspace(0,5,100)));
ylabel('Target');
xlabel('Input');
legend({'Training set' 'Validation set' 'Test set' 'Real function'});

%% Training
for order = 0:9
    lin_model{order+1} = fitlm(x_trai, y_trai, ['poly' num2str(order)]);
    y_pred = predict(lin_model{order+1}, x_trai);
    MSE(order+1) = mean((y_pred - y_trai).^2);
end

figure();
plot(MSE);
title('Only training');
xlabel('Model parameters');
ylabel('MSE');

%% Test
for order = 0:9
    y_pred = predict(lin_model{order+1}, x_test);
    MSE_test(order+1) = mean((y_pred - y_test).^2);
end

figure();
plot(MSE);
hold on;
plot(MSE_test);
legend({'Training MSE' 'Test MSE'});
title('Training and test error');
xlabel('Model parameters');
ylabel('MSE');

%% Validation
for order = 0:9
    y_pred = predict(lin_model{order+1}, x_vali);
    MSE_vali(order+1) = mean((y_pred - y_vali).^2);
end

figure()
plot(MSE);
hold on;
plot(MSE_test);
plot(MSE_vali);
[y_min, x_min] = min(MSE_vali);
plot(x_min,y_min,'x');
legend({'Training MSE' 'Test MSE' 'Validation MSE'});
xlabel('Model parameters');
ylabel('MSE');

%% Crossvalidation
tic
k_fold = 30;

x_cross = [x_trai; x_vali];
y_cross = [y_trai; y_vali];
n_cross = n_trai + n_vali;

for order = 0:9
    MSE_cross(order+1) = 0;
    % Divide data
    for kk = 1:k_fold
        ind_vali = 1 + round(n_cross * (kk - 1) / k_fold ) : ...
            round(n_cross * kk / k_fold );
        
        x_vali = x_cross(ind_vali);
        y_vali = y_cross(ind_vali);
        x_trai = x_cross; x_trai(ind_vali) = [];
        y_trai = y_cross; y_trai(ind_vali) = [];
        
        % Fit model
        model = fitlm(x_trai, y_trai, ['poly' num2str(order)]);
        y_pred = predict(model, x_vali);
        MSE_cross(order+1) = MSE_cross(order+1) + ...
            mean((y_pred - y_vali).^2);
    end
    MSE_cross(order+1) = MSE_cross(order+1) / k_fold;
    
    %predfun = @(x_t,y_t,x_v)(eval_fitlm(x_t, y_t, x_v, order));
    %mmse(order+1) = crossval('mse',x,y,'Predfun',predfun,'kfold',k_fold);
    
end
toc

figure()
h(1) = plot(MSE);
hold on;
h(2) = plot(MSE_test);
h(3) = plot(MSE_vali);
[y_min, x_min] = min(MSE_vali);
plot(x_min,y_min,'x');
h(4) = plot(MSE_cross);
[y_min, x_min] = min(MSE_cross);
plot(x_min,y_min,'x');
%h(5) = plot(mmse);
legend(h,{'Training MSE' 'Test MSE' 'Validation MSE' ...
    'Crossvalidation MSE'},'location','northwest');
xlabel('Model parameters');
ylabel('MSE');
