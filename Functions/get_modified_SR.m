function mod_SR = get_modified_SR(x, p, logRet, rf_rate)
%
% INPUT
% x : portfolio's weights
% p : confidence interval of the VaR
% logRet : log-return of the assets
% rf_rate : risk free rate
%
% OUTPUT
% mod_SR : modified Sharpe Ratio

pRet = x'*logRet';
VaR = - quantile(pRet,1-p);
expRet = mean(logRet);
mod_SR = ((x')*expRet'-rf_rate)/VaR;

end