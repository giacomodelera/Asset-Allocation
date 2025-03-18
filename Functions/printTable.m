function t = printTable(ptfNames,ptfMatrix, logRet, weightMKT, rfr)
%
% INPUT
%
% ptfNames : array of string of portfolios' name
% ptfMatrix: matrix where each column is a portfolio
% logRet : log-return of the assets
% weightMKT : market portfolio
% rfr : Risk free rate
%
% OUTPUT
%
% t : table with metrics
%

t = table( 'Size', [0 9], ...
         'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'}, ...
          'VariableNames',{'PortfolioName','AnnualReturn','AnnualVolatility','AnnualSharpeRatio', 'MaxDD', 'DailyAlpha', 'Beta', 'HerfindahlIndex', 'maxWeight'});

for i = 1:length(ptfNames)
    [annRet, annVol, annSR, MDD, alpha, beta, HI, maxWeight] = getMetrics(ptfMatrix(:,i), logRet, weightMKT,rfr);
    t(i,:) = {ptfNames(i), annRet, annVol, annSR, MDD, alpha, beta, HI, maxWeight};
end

end