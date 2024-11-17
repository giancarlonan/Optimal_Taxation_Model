function cstar = cons(wage,lambda,m, gamma, lstar)

% Optimal individual consumption under the HSV progressive taxation.
% Simplified 
cstar = (lambda*(wage*lstar)^(1-gamma)+m);  
