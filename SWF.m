function socialwelfare = SWF(wagedata,gamma,N,weights, lambda)

% calculating social welfare function - the objective function of
% the optimal income taxation problem. 
% Inputs: 
% 1. wagedata (of N individuals)
% 2. Progressivity Parameter, gamma.
% 3. N (number of individuals)
% 4. weights (of N individuals)
% 5. Scale Parameter for taxation, lambda. 

socialwelfare = 0;
for i = 1:N    
    wage = wagedata(i);      % wage for individual i
    % utility for each individual using the quasi-linear utility function.
    indutil = cons(wage,lambda,transfers(wagedata,gamma,lambda,N), gamma, labor(wage, gamma, lambda)) - (1/2)*(labor(wage,gamma,lambda)^2); 
    socialwelfare = socialwelfare + weights(i)*indutil;   %  Adding to the social welfare with the WEIGHTED individual utility.                                  
end


