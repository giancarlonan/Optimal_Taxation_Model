function lstar = labor(wage,gamma,lambda)

% Optimal Labour supply under the HSV progressive taxation
lstar = (lambda*(1-gamma)*(wage^(1-gamma)))^(1/(1+gamma)); 