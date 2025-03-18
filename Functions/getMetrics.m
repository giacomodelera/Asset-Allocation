function [annRet, annVol, annSR, MDD, alpha, beta, HI, maxWeight] = getMetrics(weights, logRet, weightMKT,rfr)
%
% INPUT
%
% weights : portfolio weights
% logRet : log-return of the assets
% weightMKT : market portfolio
% rfr : Risk free rate
%
% OUTPUT
%
% annRet : annual return
% annVol : annual volatility
% annSR : annual Sharpe ratio
% MDD : maximum drawdown
% alpha : extra return from the capm
% beta : "correlation" with the market
% HI : Herfindahl index

%var = var(logRet);
%std = std(logRet);
V = cov(logRet);
expRetAsset = mean(logRet,1);
N = 252; %Trading days

ptfRet = logRet*weights;
ptfRetMKT = logRet*weightMKT;

% Expected return (annual)
expRet = expRetAsset*weights * N; 
expRetMKt = expRetAsset*weightMKT*N;

% ANNUAL RETURN
annRet = sum(ptfRet);

% ANNUAL VOLATILITY
annVol = std(ptfRet) * sqrt(N);

% SHARPE RATIO
annSR = (annRet - rfr)/annVol;

% MAXIMUM DRAWDOWN (daily)
NAV = exp(cumsum(ptfRet));
peak = cummax(NAV);

MDD = max((peak - NAV)./peak);

% ALPHA E BETA
capm = fitlm(ptfRetMKT,ptfRet);

alpha = capm.Coefficients.Estimate(1);
beta = capm.Coefficients.Estimate(2);

% HERFINDAHL INDEX
HI = sum(weights.^2);

% MAX WEIGHT

maxWeight = max(weights);

end