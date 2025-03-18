function [risk_frontier, ret_frontier, MVP_ptf, MSR_ptf, risk_min_var, ret_min_var, risk_max_sharpe, ret_max_sharpe]=FrontierConstrains(VarCovarMatrix, ptfRet, logRetStocks, numPointFrontier, numAssets,A,b,nonLinContr,options)
    
    %
    % This function computes the portfolio frontier using the constrains
    % specified in the ExamA_2024.pdf
    %
    % INPUT:
    % VarCovarMatrix : Variance-Covariance Matrix of the log-returns
    % ptfRet : Optimal allocation according to the Frontier with standard
    % assumption
    % logRetStocks : Matrix of log-return time series
    % numPointFrontier : Desidered number of points on the frontier
    % numAsset : number of assets
    % A,b, nonLinContr, options : Parameters for the "fmincon(...)" command
    %
    % OUTPUT
    %
    % risk_frontier, ret_frontier : Portfolio frontier
    % MVP_ptf : minimum variance portfolio
    % MSR_ptf : maximum sharpe ratio portfolio
    %

    fun = @(x)x'*VarCovarMatrix*x;
    ret_ = linspace(min(ptfRet), max(ptfRet),numPointFrontier);
    NumAssets = 16;
    x0 = rand(1,numAssets)';
    x0 = x0/sum(x0);
    lb = zeros(1,numAssets);
    ub = ones(1,numAssets);
    risk_frontier = zeros(1, length(ret_));
    ret_frontier = zeros(1, length(ret_));

    w_opt = zeros(NumAssets,numPointFrontier);
    sharpe_ratio = zeros(numPointFrontier,1);

    %h = waitbar(0, 'Working...');

    for i = 1:length(ret_)
        r = ret_(i);
        Aeq = [ones(1,NumAssets); mean(logRetStocks)]; 
        beq =[1; r];
        % find optimal w, minimizing volatility
        w_opt(:,i)= fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonLinContr, options);
        
        risk_frontier(i) = sqrt(w_opt(:,i)'*VarCovarMatrix*w_opt(:,i));
        ret_frontier(i)= r; %w_opt'*exp_ret';
        sharpe_ratio(i) = r/risk_frontier(i);
        %waitbar(i / length(ret_), h, sprintf('Progress: %d%%', round(i / length(ret_) * 100)));

    end

    %close(h);

    % Minimum Variance Portfolio
    [~,idx] = min(risk_frontier);
    MVP_ptf = w_opt(:,idx);
    risk_min_var = risk_frontier(idx);
    ret_min_var = ret_frontier(idx);
    
    % Maximum Sharpe Ratio portfolio
    [~,idx2] = max(sharpe_ratio);
    MSR_ptf = w_opt(:,idx2);
    risk_max_sharpe = risk_frontier(idx2);
    ret_max_sharpe = ret_frontier(idx2);

end
