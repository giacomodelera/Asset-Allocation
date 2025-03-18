function [c,ceq] = constrainPCA(w, VarCovarMatrix, threshold)
%
% INPUT
%
% w: portfolio weights
% VarCovarMatrix : Variance-Covariance matrix con log-returns
% threshold : target volatility
%
% OUTPUT
%
% c : inequality constrains;
% ceq : equality constrains;

    c = [];
    ceq = [w'*VarCovarMatrix*w - threshold];
end