%% EXAM A - GROUP 2

% Bechi Carlo, Bencini Margherita, Ciotti Alessandro, Delera Giacomo, Riondato Giovanni

%% Load Data

clear all
close all
clc

rng(0)

warning('off','all');

table_prices = readtable("prices.xlsx");
table_capitalization = readtable("capitalizations.xlsx");

% trasforming prices from table to timetable
dt = table_prices(:,1).Variables;
prices=table_prices(:,2:end).Variables;
names=table_prices.Properties.VariableNames(2:end);

% we buld the time table
timetable_prices=array2timetable(prices, 'RowTimes', dt, 'VariableNames', names);

% Selecting the data from 2023
start_date=datetime('01/01/2023', 'InputFormat', 'dd/MM/yyyy');
end_date=datetime('31/12/2023', 'InputFormat', 'dd/MM/yyyy');
rng_=timerange(start_date, end_date,'closed');
subsample=timetable_prices(start_date:end_date,:);

values=subsample.Variables;
date=subsample.Time;

% Log returns, Variance and covariance matrix
logRet = tick2ret(values,"Method","continuous");

expLogRet = mean(logRet);
varLogRet = var(logRet);
stdLogRet = std(logRet);

standardized_Ret = (logRet - expLogRet)./ stdLogRet;  

V = cov(logRet);


%% ITEM 1 - EFFICIENT FRONTIER --> standard constrains
fprintf('\n <strong>-------------------- ITEM 1: EFFICIENT FRONTIER (standard constrains) --------------------</strong> \n');
% Portfolio object
p = Portfolio('AssetList',names);

% all weights sum to 1, no shorting, and 100% investment in risky assets).
p = setDefaultConstraints(p);
p = estimateAssetMoments(p, logRet,'missingdata',false);

pwgt = estimateFrontier(p, 100); % Estimate the weights.
[pf_Risk, pf_Retn] = estimatePortMoments(p, pwgt); % Get the risk and return.

% Minimum Variance Portfolio
[ptfA,~,~] = estimateFrontierByRisk(p,min(pf_Risk));
[pf_RiskA, pf_RetnA] = estimatePortMoments(p, ptfA);

% Maximum Sharpe Ratio Portfolio
[ptfB,~,~] = estimateMaxSharpeRatio(p);
[pf_RiskB, pf_RetnB] = estimatePortMoments(p, ptfB);

figure('Name','ITEM 1')
plot(pf_Risk,pf_Retn, 'LineWidth',2, 'DisplayName','Frontier');
hold on
grid on

% Highlight Minimum Variance Portfolio (Portfolio A)
scatter(sqrt(ptfA' * V * ptfA), mean(logRet) * ptfA, 100, 'r', 'filled', 'DisplayName', 'Portfolio A (Min Variance)');
fprintf('PtfA: standard deviation = %d, return = %d \n', sqrt(ptfA' * V * ptfA), mean(logRet) * ptfA)

% Highlight Maximum Sharpe Ratio Portfolio (Portfolio B)
scatter(sqrt(ptfB' * V * ptfB), mean(logRet) * ptfB, 100, 'g', 'filled', 'DisplayName', 'Portfolio B (Max Sharpe)');
fprintf('PtfB: standard deviation = %d, return = %d \n', sqrt(ptfB' * V * ptfB), mean(logRet) * ptfB)

title("Standard efficient frontier");
xlabel("Standard deviation")
ylabel("Return")

legend('show', 'Location', 'best');
hold off


%% ITEM 2 - EFFICIENT FRONTIER --> multiple constrains
fprintf('\n <strong>-------------------- ITEM 2: EFFICIENT FRONTIER (multiple constrains) --------------------</strong> \n');

NumAssets = 16;
pointOnFrontier = 200;

A = zeros(3,NumAssets);
b = zeros(3,1);

% total factors weight over 15%
names_factor={'Momentum','Value','Growth','Quality','LowVolatility'};
idx=ismember(names,names_factor);
A(1,idx) = -1;
b(1) = -0.15;

%cyclical & defensive over 40%
names_cycl_def={'Financials','HealthCare','ConsumerDiscretionary','Industrials','ConsumerStaples','Utilities','RealEstate','Materials'};
idx=ismember(names,names_cycl_def);
A(2,idx) = -1;
b(2) = -0.4;

%cyclical & defensive under 70%
A(3,idx) = 1;
b(3) = 0.7;

%portfolio constrains
p = Portfolio('AssetList',names);
p = estimateAssetMoments(p, logRet,'missingdata',false);
p = setDefaultConstraints(p);
p = addInequality(p,A,b);
p = setBounds(p,1e-4,1);
p = setMinMaxNumAssets(p,12,16);


pwgt = estimateFrontier(p, 100); % Estimate the weights.
[pf_Risk, pf_Retn] = estimatePortMoments(p, pwgt); % Get the risk and return.

% Minimum Variance Portfolio
[ptfC,~,~] = estimateFrontierByRisk(p,min(pf_Risk));
[pf_RiskC, pf_RetnC] = estimatePortMoments(p, ptfC);

% Maximum Sharpe Ratio Portfolio
[ptfD,~,~] = estimateMaxSharpeRatio(p);
[pf_RiskD, pf_RetnD] = estimatePortMoments(p, ptfD);

figure('Name','ITEM 2')
plot(pf_Risk,pf_Retn, 'LineWidth',2, 'DisplayName', 'Efficient Frontier with Constraints');
hold on
grid on

scatter(pf_RiskC, pf_RetnC, 100, 'r', 'filled', 'DisplayName', 'Portfolio C (Min Variance)');
fprintf('PtfC: standard deviation = %d, return = %d \n', pf_RiskC, pf_RetnC)

scatter(pf_RiskD, pf_RetnD, 100, 'g', 'filled', 'DisplayName', 'Portfolio D (Max Sharpe)');
fprintf('PtfD: standard deviation = %d, return = %d \n', pf_RiskD, pf_RetnD)

title("Efficient frontier with multiple constrains");
xlabel("Standard deviation")
ylabel("Return")

legend('show', 'Location', 'best');
hold off

%% ITEM 3 ROBUST FRONTIER --> standard constrains
fprintf('\n <strong>-------------------- ITEM 3: ROBUST FRONTIER (standard constrains) --------------------</strong> \n');

N = 100; %Number of simulation
M = 100; %Point on the frontier
NumAssets = 16;

mean_logRet = mean(logRet);
cov_logRet = cov(logRet);
sample_size = length(logRet);

vol_PtfSim = zeros(M,N);
ret_PtfSim = zeros(M,N);
weights = zeros(NumAssets,M,N);

p = Portfolio('AssetList', names);
p = setDefaultConstraints(p);

w_mvp = zeros(NumAssets,N);
w_msr = zeros(NumAssets,N);
h = waitbar(0, 'Working...');
for i = 1:N
    randRet = mvnrnd(mean_logRet,cov_logRet,sample_size);
    new_ExpRet = mean(randRet);
    %new_Cov = iwishrnd(V,NumAssets);
    new_Cov = cov(randRet);
    
    Psim = setAssetMoments(p, new_ExpRet, new_Cov);
    w_sim = estimateFrontier(Psim, M);
    weights(:,:,i) = w_sim;
    [ptf_volSim, ptf_retSim] = estimatePortMoments(Psim,w_sim);
    ret_PtfSim(:,i) = ptf_retSim;
    vol_PtfSim(:,i) = ptf_volSim;
    
    % Minimum Variance Portfolio
    [w_mvp(:,i),~,~] = estimateFrontierByRisk(Psim,min(ptf_volSim));
    
    % Maximum Sharpe Ratio Portfolio
    [w_msr(:,i),~,~] = estimateMaxSharpeRatio(Psim);
    waitbar(i / N, h, sprintf('Progress: %d%%', round(i / N * 100)));
end

close(h);

w_opt_rob = mean(weights,3);
ret_frontier = expLogRet * w_opt_rob;
vol_frontier = zeros(1,M);
for i = 1 : M
    vol_frontier(i) = sqrt(w_opt_rob(:,i)' * V * w_opt_rob(:,i));
end

[~,idx] = min(vol_frontier);
ptfE = w_opt_rob(:,idx);

sharpe_frontier = ret_frontier./vol_frontier;
[~,idx] = max(sharpe_frontier);
ptfF= w_opt_rob(:,idx);

figure('Name','ITEM 3 (standard constrains)')
plot(vol_frontier, ret_frontier, 'LineWidth',2, 'DisplayName','Robust Frontier');
hold on

scatter(sqrt(ptfE'*V*ptfE), mean(logRet)*ptfE, 100, 'r', 'filled', 'DisplayName', 'Portfolio E (Min Variance)');
fprintf('PtfE: standard deviation = %d, return = %d \n', sqrt(ptfE'*V*ptfE), mean(logRet)*ptfE)

scatter(sqrt(ptfF'*V*ptfF), mean(logRet)*ptfF, 100, 'g', 'filled', 'DisplayName', 'Portfolio F (Max Sharpe)');
fprintf('PtfF: standard deviation = %d, return = %d \n', sqrt(ptfF'*V*ptfF), mean(logRet)*ptfF)

title("Robust efficient frontier");
xlabel("Standard deviation");
ylabel("Return");
legend('show', 'Location', 'best');
grid on
hold off

%% ITEM 3 ROBUST FRONTIER --> multiple constrains

N = 10; % number of simulation
M = 25; %Point on the frontier
NumAssets = 16;
vol_PtfSim = zeros(M,N); ret_PtfSim = zeros(M,N);
weightsSim = zeros(M,NumAssets,N);
w_mvpSim = zeros(NumAssets,N); w_msrSim = zeros(NumAssets,N);


A = zeros(3,NumAssets);
b = zeros(3,1);
% total factors weight over 15%
A(1,12:16) = -1; b(1) = -0.15;
%cyclical & defensive over 40%
idx = [4,2,11,10,6,7,9,3]; A(2,idx) = -1; b(2) = -0.4;
%cyclical & defensive under 70%
A(3,idx) = 1; b(3) = 0.7;

nonlinControl = @(x) NumAssetConstrain(x, 12,16) ;
options = optimoptions('fmincon','HessianApproximation','lbfgs','Algorithm','sqp','StepTolerance',1e-12, 'Display','off');

h = waitbar(0, 'Working...');
for i = 1:N
    randRet = mvnrnd(mean_logRet,cov_logRet,sample_size);
    new_ExpRet = mean(randRet);
    %new_Cov = iwishrnd(V,NumAssets);
    new_Cov = cov(randRet);

    p = Portfolio('AssetList', names);
    p = setDefaultConstraints(p);
    Psim = setAssetMoments(p, new_ExpRet, new_Cov);
    w_sim = estimateFrontier(Psim, M);
    [~, pf_ret] = estimatePortMoments(Psim,w_sim);

    [vol_PtfSim(:, i), ret_PtfSim(:,i), w_mvpSim(:,i), w_msrSim(:,i)] = FrontierConstrains(V, pf_ret, logRet, M, NumAssets,A,b,nonlinControl,options);
    waitbar(i / N, h, sprintf('Progress: %d%%', round(i / N * 100)));
end

close(h);

% Final frontier
figure
vol_frontier = mean(vol_PtfSim,2);
ret_frontier = mean(ret_PtfSim,2);
plot(vol_frontier, ret_frontier, 'LineWidth', 4)
hold on
title("Frontier with robust method - non linear constrains");
xlabel("Standard deviation");
ylabel("Return");
grid on

% minimum variance portfolio with standard constrain
ptfG = mean(w_mvpSim,2);
% maximum sharpe ratio portfolio with standard constrain
ptfH = mean(w_msrSim,2);

scatter(sqrt(ptfG'*V*ptfG), mean(logRet)*ptfG, 100, 'r', 'filled', 'DisplayName', 'Portfolio G (Min Variance)');
scatter(sqrt(ptfH'*V*ptfH), mean(logRet)*ptfH, 100, 'g', 'filled', 'DisplayName', 'Portfolio H (Max Sharpe)');

hold off

%% ITEM 3 ROBUST FRONTIER --> multiple constrains with portfolio object
fprintf('\n <strong>-------------------- ITEM 3: ROBUST FRONTIER (multiple constrains) --------------------</strong> \n');

N = 100; % number of simulation
M = 25; %Point on the frontier
NumAssets = 16;

mean_logRet = mean(logRet);
cov_logRet = cov(logRet);
sample_size = length(logRet);

vol_PtfSim = zeros(M,N);
ret_PtfSim = zeros(M,N);
weights = zeros(NumAssets,M,N);

p = Portfolio('AssetList', names);
p = setDefaultConstraints(p);

w_mvp = zeros(NumAssets,N);
w_msr = zeros(NumAssets,N);

h = waitbar(0, 'Working...');
for i = 1:N
    randRet = mvnrnd(mean_logRet,cov_logRet,sample_size);
    new_ExpRet = mean(randRet);
    new_Cov = cov(randRet);
    %new_Cov = iwishrnd(V,NumAssets);

    Psim = setAssetMoments(p,new_ExpRet, new_Cov);
    Psim = addInequality(Psim,A,b);
    Psim = setBounds(Psim,1e-4,1);
    Psim = setMinMaxNumAssets(Psim,12,16);
    w_sim = estimateFrontier(Psim, M);
    weights(:,:,i) = w_sim;
    [ptf_volSim, ptf_retSim] = estimatePortMoments(Psim,w_sim);
    ret_PtfSim(:,i) = ptf_retSim;
    vol_PtfSim(:,i) = ptf_volSim;
    waitbar(i / N, h, sprintf('Progress: %d%%', round(i / N * 100)));

end

close(h);

w_opt_rob = mean(weights,3);
ret_frontier = expLogRet * w_opt_rob;
vol_frontier = zeros(1,M);
for i = 1 : M
    vol_frontier(i) = sqrt(w_opt_rob(:,i)' * V * w_opt_rob(:,i));
end

[~,idx] = min(vol_frontier);
ptfG = w_opt_rob(:,idx);

sharpe_frontier = ret_frontier./vol_frontier;
[~,idx] = max(sharpe_frontier);
ptfH= w_opt_rob(:,idx);

figure('Name','ITEM 3 (multiple constrains)')
plot(vol_frontier, ret_frontier, 'LineWidth',2, 'DisplayName','Robust frontier with multiple constrains');
hold on

scatter(sqrt(ptfG'*V*ptfG), mean(logRet)*ptfG, 100, 'r', 'filled', 'DisplayName', 'Portfolio G (Min Variance)');
fprintf('PtfG: standard deviation = %d, return = %d \n', sqrt(ptfG'*V*ptfG), mean(logRet)*ptfG)


scatter(sqrt(ptfH'*V*ptfH), mean(logRet)*ptfH, 100, 'g', 'filled', 'DisplayName', 'Portfolio H (Max Sharpe)');
fprintf('PtfH: standard deviation = %d, return = %d \n', sqrt(ptfH'*V*ptfH), mean(logRet)*ptfH)


title("Robust efficient frontier with Multiple Constrains");
xlabel("Standard deviation");
ylabel("Return");
legend('show', 'Location', 'best');
grid on
hold off
toc
%% ITEM 4 - PORTFOLIO FRONTIER, BLACK LITTERMAN
fprintf('\n <strong>-------------------- ITEM 4: PORTFOLIO FRONTIER with BLACK LITTERMAN --------------------</strong> \n');

v = 2; % 2 views
tau = 1/length(logRet);
NumAssets = 16;

P = zeros(v,NumAssets);
q = zeros(v,1);
Omega = zeros(v);

%View 1: cyclical outperform defensive by 2%
P(1, names == "ConsumerDiscretionary") = 1/5;
P(1, names == "Financials") = 1/5;
P(1, names == "Materials") = 1/5;
P(1, names == "RealEstate") = 1/5;
P(1, names == "Industrials") = 1/5;

P(1, names == "ConsumerStaples") = -1/3;
P(1, names == "Utilities") = -1/3;
P(1, names == "HealthCare") = -1/3;

q(1) = 0.02;

%View 2: Value outperforms Growth by 1%
P(2, names == "Value")=1;
P(2, names == "Growth")=-1;
q(2) = 0.01;

Omega(1,1) = tau.*P(1,:)*V*P(1,:)';
Omega(2,2) = tau.*P(2,:)*V*P(2,:)';

year2day = 1/252;
q = q*year2day;
Omega = Omega*year2day;

caps = table_capitalization(1,2:end).Variables;
w_MKT = caps'/sum(caps);
lambda = 1.2;
mu_mkt = lambda.*V*w_MKT;
C = tau.*V;

% mean and covariance of Black Litterman
muBL = inv( inv(C) + P'*inv(Omega)*P)*(P'*inv(Omega)*q + inv(C)*mu_mkt);
covBL = inv(P'*inv(Omega)*P + inv(C));

portBL = Portfolio('NumAssets',NumAssets,'Name','MV with BL');
portBL = setDefaultConstraints(portBL);
portBL = setAssetMoments(portBL, muBL, V + covBL);

N = 250; %Number of point on the frontier
wptf_BL = estimateFrontier(portBL,250);
[riskBL, retBL] = estimatePortMoments(portBL,wptf_BL);

% MINIMUM VARIANCE PORTFOLIO              
[ptfI,~,~] = estimateFrontierByRisk(portBL,min(riskBL));
[pf_RiskI, pf_RetnI] = estimatePortMoments(portBL, ptfI);  

% MAX SHARPE RATIO PORTFOLIO
[ptfL,~,~] = estimateMaxSharpeRatio(portBL);
[pf_RiskL, pf_RetnL] = estimatePortMoments(portBL, ptfL);

figure('Name', 'ITEM 4')
plot(riskBL,retBL, 'LineWidth',2, 'DisplayName','Black-Litterman frontier')
hold on
grid on

scatter(pf_RiskI, pf_RetnI, 100, 'r', 'filled', 'DisplayName', 'Portfolio I (Min Variance)');
fprintf('PtfI: standard deviation = %d, return = %d \n', pf_RiskI, pf_RetnI)


scatter(pf_RiskL, pf_RetnL, 100, 'g', 'filled', 'DisplayName', 'Portfolio L (Max Sharpe)');
fprintf('PtfL: standard deviation = %d, return = %d \n', pf_RiskL, pf_RetnL)


title("Efficient Frontier with Black-Litterman model")
xlabel("Standard Deviation")
ylabel("Return")
legend('show', 'Location', 'best');
hold off

%% ITEM 5 - MAXIMUM DIVERSIFIED PTF & MAXIMUM ENTROPY
fprintf('\n <strong>-------------------- ITEM 5: MAXIMUM DIVERSIFIED PTF & MAXIMUM ENTROPY --------------------</strong> \n');

NumAssets = 16;

% sum(w) = 1
A_eq = ones(1, NumAssets);
b_eq = 1;

%no short selling
lb = zeros(NumAssets,1);
ub = ones(NumAssets,1);

%  weights of factor indices are 0.05 ≤ wi ≤ 0.1
A = [zeros(11,NumAssets);
    double(names == "Momentum");
    double(names == "Momentum")*-1;
    double(names == "Value");
    double(names == "Value")*-1;
    double(names == "Growth");
    double(names == "Growth")*-1;
    double(names == "Quality");
    double(names == "Quality")*-1;
    double(names == "LowVolatility");
    double(names == "LowVolatility")*-1;
    ];

b = [zeros(11,1);
    0.1;
    -0.05;
    0.1;
    -0.05;
    0.1;
    -0.05;
    0.1;
    -0.05;
    0.1;
    -0.05;
    ];

%constrain wrt capitalization weighted ptf
caps = table_capitalization(1,2:end).Variables;
w_MKT = caps'/sum(caps);

nonLinConst = @(x) ConstrainPoint5(x,w_MKT,0.5);

% initial condition
x0 = ones(NumAssets,1)/NumAssets;

% MAXIMUM DIVERSIFIED PORTFOLIO
f = @(w) -log( (w'*stdLogRet')/sqrt(w'*V*w));

options = optimoptions('fmincon','Algorithm','sqp','StepTolerance',1e-12,"EnableFeasibilityMode",true, 'MaxFunctionEvaluations',10000);
ptfM = fmincon(f,x0,A,b,A_eq,b_eq,lb,ub,nonLinConst,options);
fprintf('PtfM: standard deviation = %d, return = %d \n', sqrt(ptfM'*V*ptfM), mean(logRet)*ptfM)

% MAXIMUM ENTROPY (in risk contribution) PORTFOLIO
theta = @(w) abs(w.*(V*w)) / sum(abs(w.*(V*w)));

f = @(w) theta(w)'*log(theta(w));

options = optimoptions('fmincon','StepTolerance',1e-12);
ptfN = fmincon(f,x0,A,b,A_eq,b_eq,lb,ub,nonLinConst, options);
fprintf('PtfN: standard deviation = %d, return = %d \n', sqrt(ptfN'*V*ptfN), mean(logRet)*ptfN)

%% ITEM 6 - PCA
fprintf('\n <strong>-------------------- ITEM 6: PORTFOLIO FRONTIER with PCA --------------------</strong> \n');

% Apply PCA on the standardized log returns
[factorLoading, factorRetn, latent, r, explained, mu] = pca(standardized_Ret);

% Calculate cumulative explained variance
CumExplVar = cumsum(explained);

% Find the minimum number of components needed to explain at least 85% of the variance
k = find(CumExplVar > 85, 1, 'first');
fprintf('Components needed to explain at least 85%% of the variance: %d\n', k);

% Retain only the top k components
factorLoading = factorLoading(:, 1:k);
factorRetn = factorRetn(:, 1:k);

% Compute the covariance matrix of the factor returns
covarFactor = cov(factorRetn);

% Calculate the total variance and the explained variance for each component
TotVar = sum(latent); % Total variance across all components
ExplainedVar = latent(1:k) / TotVar; % Proportion of variance explained by each component

% Create a list of component indices
n_list = linspace(1, k, k);
CumExplainedVar = zeros(1, size(n_list, 2));

% Compute the cumulative explained variance for the first n components
for i = 1:size(n_list, 2)
    n = n_list(i);
    % Custom function to calculate cumulative explained variance
    CumExplainedVar(1, i) = getCumulativeExplainedVar(latent, n);
end

figure('Position', [200, 200, 1000, 400], 'Name', 'ITEM 6');

% Plot the explained variance for each principal component
subplot(1,2,1);
bar(n_list, ExplainedVar);
xlabel('Principal Components');
ylabel('Percentage of Explained Variances');
title('Percentage of Explained Variances for each Principal Component');

% Plot the cumulative explained variance as a function of components
subplot(1,2,2);
plot(n_list, CumExplainedVar, 'm');
hold on;
scatter(n_list, CumExplainedVar, 'm', 'filled');
grid on;
xlabel('Total number of Principal Components');
ylabel('Percentage of Explained Variances');
title('Total Percentage of Explained Variances for the first n-components');

% Reconstruct asset returns using selected principal components
% Reconstructions include the effect of variance explained by selected components.
reconReturn = (factorRetn * factorLoading') .* stdLogRet + expLogRet;
unexplainedRetn = logRet - reconReturn;

% The unexplained returns represent asset-specific risks (diagonal covariance D)
unexplainedCovar = diag(cov(unexplainedRetn));
D = diag(unexplainedCovar);

% Compute the total asset covariance matrix
covarAsset = factorLoading * covarFactor * factorLoading' + D;

% Optimization to compute the portfolio weights
% Define the objective function (negative of expected return adjusted for risk)
func = @(x) -((expLogRet * x) - ((factorLoading' * x)' * covarFactor * (factorLoading' * x) + x' * D * x));

% Define the objective function (negative of expected)
%func = @(x) -(expLogRet * x);

% Initial guess for portfolio weights
x0 = rand(size(logRet, 2), 1);
x0 = x0 ./ sum(x0); % Normalize to ensure weights sum to 1

% Set bounds, equality constraints, and inequality constraints for weights
lb = zeros(1, size(logRet, 2)); % No short-selling allowed
ub = ones(1, size(logRet, 2)); % Maximum weight of 1 per asset
Aeq = ones(1, size(logRet, 2));
beq = 1; % Sum of weights must equal 1

% Define the nonlinear constraint to limit total portfolio risk
% 0.65 is the target volatility
nonlcon = @(x) deal(sqrt((factorLoading' * x)' * covarFactor * (factorLoading' * x) + x' * D * x) - 0.65, []);

% Set optimization options
options = optimoptions('fmincon', 'Display', 'final', 'Algorithm', 'sqp', 'MaxIterations', 100000);

% Perform optimization to find the optimal portfolio weights
[w_opt, fval] = fmincon(func, x0, [], [], Aeq, beq, lb, ub, nonlcon, options);

% Compute factor exposures of the optimized portfolio
wf = (factorLoading' * w_opt);

% Calculate the portfolio volatility
P_vol = sqrt((factorLoading' * w_opt)' * covarFactor * (factorLoading' * w_opt) + w_opt' * D * w_opt);


ptfP = w_opt;
fprintf('PtfP: standard deviation = %d, return = %d \n', sqrt(ptfP'*V*ptfP), mean(logRet)*ptfP)

%% ITEM 7 - VaR
fprintf('\n <strong>-------------------- ITEM 7: PORTFOLIO FRONTIER with VaR --------------------</strong> \n');

options = optimoptions('fmincon', ...
    'OptimalityTolerance', 1e-12, ...
    'StepTolerance', 1e-12, ...
    'ConstraintTolerance', 1e-12, ...
    'Display','off');

%Conflevel = [0.95,0.99];
Conflevel = 0.99;
rf_rate = 0;

% setting standard constrains
%x0 = ones(size(logRet,2),1);
%x0 = x0./sum(x0);
lb = zeros(1, size(logRet,2))+0.001; 
ub = ones(1, size(logRet,2));
Aeq = ones(1, size(logRet,2));
beq = 1;

%ptfQ = zeros(size(logRet,2),2);
%fun_95 = @(x) -get_modified_SR(x, Conflevel(1), logRet, rf_rate);
%fun_99 = @(x) -get_modified_SR(x, Conflevel(2), logRet, rf_rate);
fun_99 = @(x) -get_modified_SR(x, Conflevel, logRet, rf_rate);

% computing portfolio Q
%ptfQ(:,1) = fmincon(fun_95, x0, [],[], Aeq, beq, lb, ub,[],options);
%ptfQ(:,2) = fmincon(fun_99, x0, [],[], Aeq, beq, lb, ub,[],options);

% considered only with 99% confidence level
%ptfQ = ptfQ(:,2);

% MonteCarlo approach
Nsim = 500;
ptfs = zeros(NumAssets,Nsim);
ratios = zeros(1,Nsim);

h = waitbar(0, 'Working...');
rng(0)
for i=1:Nsim
    x0 = rand(NumAssets,1);
    x0 = x0./sum(x0);
    ptfs(:,i) = fmincon(fun_99, x0, [],[], Aeq, beq, lb, ub,[],options);

    ratios(i) = get_modified_SR(ptfs(:,i), Conflevel, logRet, rf_rate);
    waitbar(i / Nsim, h, sprintf('Progress: %d%%', round(i / Nsim * 100)));
end

close(h);

% max modified sharpe ratio ptf among all the simulations
[~,idx] = max(ratios);
ptfQ = ptfs(:,idx);
fprintf('PtfQ: standard deviation = %d, return = %d \n', sqrt(ptfQ'*V*ptfQ), mean(logRet)*ptfQ)



%% ITEM 8 - VALUATION OF THE PORTFOLIOs OVER 2023
fprintf('\n <strong>-------------------- VALUATION OF THE PORTFOLIOs OVER 2023 --------------------</strong> \n');

caps = table_capitalization(1,2:end).Variables;
w_MKT = caps'/sum(caps); % MARKET PORTFOLIO
NumAssets = 16;
rf_rate = 0.03;

w_eqWeighted = ones(NumAssets,1)/NumAssets; % Equally weighted portfolio

ptfNames = ["Equally weighted", "ptfA", "ptfB", "ptfC", "ptfD" , "ptfE", "ptfF", "ptfG", "ptfH", "ptfI", "ptfL", "ptfM", "ptfN", "ptfP", "ptfQ"];
ptfMatrix = [w_eqWeighted ptfA ptfB, ptfC, ptfD, ptfE, ptfF, ptfG, ptfH, ptfI, ptfL, ptfM, ptfN, ptfP, ptfQ];

for i=1:length(ptfNames)
    show_pie(ptfMatrix(:,i), ptfNames(i))
end

table2023 = printTable(ptfNames,ptfMatrix,logRet,w_MKT,rf_rate)

%%
hm_table = printTable_2(ptfNames,ptfMatrix,logRet,w_MKT,rf_rate);

%% PART B - VALUATION OF THE PORTFOLIOs OVER 2024
fprintf('\n <strong>-------------------- VALUATION OF THE PORTFOLIOs OVER 2024 --------------------</strong> \n');

% Selecting the data from 2023
start_date=datetime('02/01/2024', 'InputFormat', 'dd/MM/yyyy');
end_date=datetime('25/10/2024', 'InputFormat', 'dd/MM/yyyy');
rng_=timerange(start_date, end_date,'closed');
subsample2024=timetable_prices(start_date:end_date,:);

values2024=subsample2024.Variables;
date2024=subsample2024.Time;

% Log returns, Variance and covariance matrix
logRet2024 = tick2ret(values2024,"Method","continuous");


ptfNames = ["Equally weighted", "ptfA", "ptfB", "ptfC", "ptfD" , "ptfE", "ptfF", "ptfG", "ptfH", "ptfI", "ptfL", "ptfM", "ptfN", "ptfP", "ptfQ"];
ptfMatrix = [w_eqWeighted ptfA ptfB, ptfC, ptfD, ptfE, ptfF, ptfG, ptfH, ptfI, ptfL, ptfM, ptfN, ptfP, ptfQ];
rf_rate = 0.03;

table2024 = printTable(ptfNames,ptfMatrix,logRet2024,w_MKT,rf_rate)

%% EQUITY CURVE

ret = values(2:end,:)./values(1:end-1,:);

n_portfolios = length(ptfNames);
equity = [] ;

colormap_ = parula(n_portfolios);

%Equity curve
for i = 1:n_portfolios
equity(:,i) = cumprod(ret*ptfMatrix(:,i));
equity(:,i) = 100.*equity(:,i)/equity(1,i);%force the equity to start from the same value
end
% Plot
dates_ = subsample.Time;
f = figure('Name','Equity 2023');
title('Equity 2023')
hold on
for i = 1:n_portfolios
    if i == 1
        plot(dates_(2:end, 1), equity(:, i), 'LineWidth', 4, 'Color', colormap_(i, :)); % Thicker line for the equally weighted
    else
        plot(dates_(2:end, 1), equity(:, i), 'LineWidth', 2, 'Color', colormap_(i, :));
    end
legend('Equally Weighted Portfolio', 'A', 'B', 'C', 'D','E','F','G','H','I','L','M','N','P','Q', 'Location', 'best')
xlabel('Date')
ylabel('Equity')
end
hold off

% portfolios
start_date_B = datetime('01/01/2024', 'InputFormat', 'dd/MM/yyyy');
end_date_B = datetime('31/12/2024', 'InputFormat', 'dd/MM/yyyy');
dates_range_B = timerange(start_date_B, end_date_B, "closed");
subsample_B = timetable_prices(dates_range_B, :);
array_assets_B = subsample_B.Variables; % array of prices
ret = array_assets_B(2:end,:)./array_assets_B(1:end-1,:);
equity = [] ;

%Equity curve
for i = 1:n_portfolios
equity(:,i) = cumprod(ret*ptfMatrix(:,i));
equity(:,i) = 100.*equity(:,i)/equity(1,i);%force the equity to start from the same value
end
% Plot
dates_ = subsample_B.Time;
f = figure('Name','Equity 2024');
title('Equity 2024')
hold on
for i = 1:n_portfolios
    if i == 1
        plot(dates_(2:end, 1), equity(:, i), 'LineWidth', 4, 'Color', colormap_(i, :)); % Thicker line for the equally weighted
    else
        plot(dates_(2:end, 1), equity(:, i), 'LineWidth', 2, 'Color', colormap_(i, :));
    end
legend('Equally Weighted Portfolio', 'A', 'B', 'C', 'D','E','F','G','H','I','L','M','N','P','Q', 'Location', 'best')
xlabel('Date')
ylabel('Equity')
end
hold off
