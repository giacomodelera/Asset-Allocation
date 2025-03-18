function [c, ceq] = ConstrainPoint5(w_ptf, w_capitalization, level)
    tollerance = 1e-6;
    %c = sum( abs (w_ptf - w_capitalization) ) - level; NOT DIFFERENTIABLE
    c =  sum ( sqrt( (w_ptf - w_capitalization).^2 + tollerance )) - level;
    ceq = [];
end