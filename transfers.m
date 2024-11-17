function m = transfers(wagedata,gamma,lambda,N)

% tax revenue from each individual i = 1,...,N using the HSV progressive
% taxation function. 
% the for loop is iterrating N times for each individual and adding their tax contribution 
% to the total tax revenue, R.

totalrevenue = 0;
for i = 1:N    
    indtax = ((wagedata(i)*labor(wagedata(i),gamma,lambda)-(lambda*(wagedata(i)*labor(wagedata(i),gamma,lambda))^(1-gamma))));     
    totalrevenue = totalrevenue + indtax;
end

% non-labor income for each individual is called transfers which is the total tax revenue, R, divided by N. 

m = totalrevenue/N;     

