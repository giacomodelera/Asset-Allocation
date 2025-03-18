function T = printTable_2(ptfNames,ptfMatrix, logRet, weightMKT, rfr)
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

metrics = {'AnnualReturn','AnnualVolatility','AnnualSharpeRatio', 'MaxDD', 'DailyAlpha', 'Beta', 'HerfindahlIndex', 'maxWeight'};
names = {'Equally weighted', 'ptfA', 'ptfB', 'ptfC', 'ptfD' , 'ptfE', 'ptfF', 'ptfG', 'ptfH', 'ptfI','ptfL', 'ptfM', 'ptfN', 'ptfP', 'ptfQ'};

for i = 1:length(names)
    [annRet, annVol, annSR, MDD, alpha, beta, HI, maxWeight] = getMetrics(ptfMatrix(:,i), logRet, weightMKT,rfr);
    data(i,:) = [annRet, annVol, annSR, MDD, alpha, beta, HI, maxWeight];
end

T = array2table(data, 'VariableNames', metrics, 'RowNames', names);
Tdata = table2array(T);
h = figure('Position', [100, 100, 1000, 500]);
title('Portfolios Metrics')
h = heatmap(metrics, names, Tdata, ...
    'ColorbarVisible', true, ...       % Show color bar
    'Colormap', parula);
h.ColorScaling = 'scaledcolumns';
title('\fontsize{16}Portfolios Metrics');
end