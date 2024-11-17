% ECON3038 Taxation and the Macroeconomy

clear;
N = 10000; % Sample Size
lambda = 0.95; % Scale Parameter 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gini coefficient in USA 2022 was 0.395 (https://data.oecd.org/inequality/income-inequality.htm)

sigma_w = 0.7339;                        % Matching the Gini coefficient by varying sigma_w

wagetemp = normrnd(0, sigma_w, [1,N]);   % Draw samples with Normal(0,sigma_w^2) with sample size = N

wagedata = exp(wagetemp);              % Take exponential such that WAGEDATA is log normal

%%%%% VISUAL GINI AND LORENZ CURVE PRE OPTIMAL TAXATION %%%%%%

pop = ones(1,N);
gini(pop,wagedata) % Gini Coeff value to ensure it matches 0.395

figure;
[gini_coefficient, lorenz_curve] = gini(pop,wagedata, true); % Visual for the Pre Tax Gini Coeff

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Specifying the optimal taxation problem %%%

% Weight assignment for each individual i = 1,...,N. 

weights = ones(1,N);     % initialization of equal weights vector

omega = 0.5;             % Omega - the threshold to determine low wage status

M = 5;                   % M - capturing how much the government cares more about low wage households (five times more)

for i = 1:N
    if (wagedata(i)<=omega)
        weights(i) = M;     % if individual i is poor according to the definition, its weight is replaced by M instead of 1.
    end
end
weights = weights/sum(weights(:));     % we normalize them so that the sum of the weights equal to one.
sum(weights(:))     % check 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Behavioural Decisions of Individuals in response to tax changes -
% optimal consumption and labor found by solving the individual utility 
% maximization problem - shown in: 

      % labor.m: labor supply as a function of wage, gamma,and lambda
      % cons.m: consumption demand as a function of wage, lambda, non-labor income (transfers), gamma,
      % and optimal labor.

%%% Individual Responses through Figures 

%Response of Labor Supply for High, Median, Low Wage Indiviudals 

gammavec_example = linspace(0, 1, N);    % Creates a vector simulating changing progressivity

% Initialize vectors to store labor supply responses 
labor_supply_low = zeros(1, length(gammavec_example));                
labor_supply_median = zeros(1, length(gammavec_example));
labor_supply_high = zeros(1, length(gammavec_example));

% Sort wagedata in order so I can call the lowest, highest, and median
% values

[wagedata_sorted, original_indices] = sort(wagedata); % Also returns the original position of these sorted values

% Identify individuals based on the sorted data 

low_wage_index = original_indices(1);                   % Position of the lowest wage inside the wagedata vector
median_wage_index = original_indices(round(N/2));       % Position of the median wage inside the wagedata vector
high_wage_index = original_indices(end);                % Position of the highest wage inside the wagedata vector

% Iterate through each gamma and compute the labor supply for each individual
for i = 1:length(gammavec_example)
    gamma = gammavec_example(i);

    % Compute labor supply for each selected individual
    labor_supply_low(i) = labor(wagedata(low_wage_index), gamma, lambda);
    labor_supply_median(i) = labor(wagedata(median_wage_index), gamma, lambda);
    labor_supply_high(i) = labor(wagedata(high_wage_index), gamma, lambda);
end

% Plotting the labor supply for selected individuals
figure;
plot(gammavec_example, labor_supply_low, 'r-', 'LineWidth', 1); hold on; % Red for low wage
plot(gammavec_example, labor_supply_median, 'g--', 'LineWidth', 1); % Green dashed for median wage
plot(gammavec_example, labor_supply_high, 'b-.', 'LineWidth', 1); % Blue dot-dash for high wage
title('Labor Supply Response for Selected Individuals vs. Gamma');
xlabel('Gamma');
ylabel('Labor Supply');
legend('Low Wage', 'Median Wage', 'High Wage', 'Location', 'Best');
hold off;

%%% Consumption Response 

% Initialize vectors to store consumption 
consdata_low = zeros(1, length(gammavec_example));                
consdata_median = zeros(1, length(gammavec_example));
consdata_high = zeros(1, length(gammavec_example));


% Iterate through each gamma and compute the consumption for each individual
for i = 1:length(gammavec_example)
    gamma = gammavec_example(i);

    % Compute consumption for each selected individual
    consdata_low(i) = cons(wagedata(low_wage_index),lambda,transfers(wagedata,gamma,lambda,N), gamma, labor(wagedata(low_wage_index), gamma, lambda));
    consdata_median(i) = cons(wagedata(median_wage_index),lambda,transfers(wagedata,gamma,lambda,N), gamma, labor(wagedata(median_wage_index), gamma, lambda));
    consdata_high(i) = cons(wagedata(high_wage_index),lambda,transfers(wagedata,gamma,lambda,N), gamma, labor(wagedata(high_wage_index), gamma, lambda));
end

% Plotting the consumption for selected individuals 

figure;
plot(gammavec_example, consdata_low, 'r-', 'LineWidth', 2); hold on; % Red for low wage
plot(gammavec_example, consdata_median, 'g--', 'LineWidth', 2); % Green dashed for median wage
plot(gammavec_example, consdata_high, 'b-.', 'LineWidth', 2); % Blue dot-dash for high wage
title('Consumption Response for Selected Individuals vs. Gamma');
xlabel('Gamma');
ylabel('Consumption');
legend('Low Wage', 'Median Wage', 'High Wage', 'Location', 'Best');
hold off;

% Visual Response of Transfers by Increasing Gamma

gammavec_example = linspace(0, 1, N);    % Create vector of gammas for visuals
transfers_result = zeros(1, length(gammavec_example));   % Initialize vector to store results 

for i = 1:length(gammavec_example)            % Iterate through each gamma and compute the transfers for each one.
    gamma = gammavec_example(i);
    transfers_m_vec= transfers(wagedata, gamma, lambda, N);    % Compute transfers using  for each value of gamma in gammavec_example. 
    transfers_result(i) = transfers_m_vec;                      % we have one transfer value, m, for each gamma value. 
end

figure;                                 % plotting figure of transfers 
plot(gammavec_example, transfers_result);
title('Transfers vs Gamma');
xlabel('Gamma');
ylabel('Government Transfers');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Social Welfare - computing social welfare by summing all indiviudals
% weighted utility from their optimal labour and consumption decisions. 
% This outlined in the SWF.m file 

%Here we are visually checking how SWF (SWF.m) would change with
% progressivity taxation parameter, gamma.
 
gammavec = 0:0.01:1;             % Vector for gamma levels: 0 to 1; in 0.01 intervals.

objective_values = zeros(size(gammavec));       % Initialize array to store objective function values

for i = 1:length(gammavec)
    gamma = gammavec(i);
    objective_values(i) = SWF(wagedata, gamma, N, weights, lambda);  % The SWF for each gamma value.
end

% Plotting relationship
figure;
plot(gammavec, objective_values);
xlabel('Progressivity Parameter (\gamma)');
ylabel('Objective Function Value');
title('Objective Function by Varying \gamma');
grid on; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to find the optimal taxation within the bounds 0 and 1. 
% Using the built-in Matlab optimization function (fminbnd). 
% This minimises the objective function so, since we want the gamma that MAXIMISES 
% the objective function, we multiply by minus one.

objval = @(gamma) -SWF(wagedata,gamma,N,weights,lambda);

opt_gamma = fminbnd(objval,0,1);       % Code that finds the optimal gamma

opt_m = transfers(wagedata,opt_gamma,lambda,N);  % Code that finds optimal transfers

totalrevenue = opt_m * N;
fprintf('Total Revenue: %f\n', totalrevenue); %Print total gov. revenue


%%%%%%%%%%%%%%%%%%%%% QUANTITATIVE ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Here I am looking to compute the consumption data, income data, and labor data under
% a PRE progressive tax scenario. I.e., progressivity = 0. 

pre_tax_lambda = 0.95;  % Lambda still equals 0.95 

pre_tax_m = transfers (wagedata, 0, pre_tax_lambda, N);        % Transfers with progressivity = 0


% Initialisation of vectors 

consdata_pretax = zeros(1,N);
labordata_pretax = zeros(1,N);
incomedata_pretax = zeros(1,N);

% For every indiviudal 1 to N, this code finds consumption, 
% labor and income under no progressive taxation.

for i = 1:N
    wage = wagedata(i);
    consdata_pretax(i) = cons(wage, pre_tax_lambda, pre_tax_m, 0, labor(wage, 0, pre_tax_lambda));
    labordata_pretax(i) = labor(wage, 0, pre_tax_lambda);
    incomedata_pretax(i) = wage*labordata_pretax(i);
end

% Aggregating the pre Tax Values

aggregate_income_pre = sum(incomedata_pretax);
aggregate_consdata_pre = sum(consdata_pretax);
aggregate_labordata_pre = sum(labordata_pretax);

% Now, here is the post-tax analysis with the recently found optimal variables - baseline
% economy. 

% Initialise vectors for each variable

post_tax_consdata = zeros(1,N);
post_tax_labordata = zeros(1,N);     
post_tax_incomedata = zeros(1,N);

% For every indiviudal 1 to N, this code finds the consumption, 
% labor, and income under OPTIMAL conditions.

for i = 1:N
    wage = wagedata(i);
    post_tax_consdata(i) = cons(wage, lambda, opt_m, opt_gamma,labor(wage, opt_gamma, lambda));
    post_tax_labordata(i) = labor(wage, opt_gamma, lambda);
    post_tax_incomedata(i) = wage*post_tax_labordata(i);
end

% Aggregate Income, Consumption, Labor 

aggregate_income_post = sum(post_tax_incomedata);
aggregate_labor_post = sum(post_tax_labordata);
aggregate_consumption_post = sum(post_tax_consdata);

% Histograms of the distribution

figure;
subplot(2,2,1)
histogram(labordata_pretax)
title('Pre-Tax Labor Data ')
subplot(2,2,2)
histogram(post_tax_labordata)
title('Post-Tax Labor Data ')
subplot(2,2,3)
histogram(consdata_pretax)
title('Pre-Tax Consumption Data ')
subplot(2,2,4)
histogram(post_tax_consdata)
title('Post-Tax Consumption Data ')

% US Economy Before any counterfactual exercises (our baseline) 

% Gini Coefficients - Pre tax

pre_tax_income_gini = gini(pop, incomedata_pretax) % Calculate pre-tax income Gini coefficient

pre_tax_consdata_gini = gini(pop, consdata_pretax) %Calculate Pre-tax Consumption Gini

pre_tax_labordata_gini = gini(pop, labordata_pretax) %calculate pre tax labor gini

% Gini Coefficients - Post tax

post_tax_income_gini = gini(pop, post_tax_incomedata)% Calculate post-tax Income Gini coefficient

post_tax_labordata_gini = gini(pop,post_tax_labordata) % Calculate Post-tax Labor Gini

post_tax_consdata_gini = gini(pop, post_tax_consdata)% Calculate Post-tax Consumption Gini


% This code finds the value of the SWF - found just to find its '%' increase. 

social_welfare_pre_tax = SWF(wagedata, 0, N, weights, pre_tax_lambda);
social_welfare_post_tax = SWF(wagedata, opt_gamma, N, weights, lambda);


%%%%%%%%%%%%%%%%% Counterfactual Analysis %%%%%%%%%%%%%%%%%

% This is the code that shows the impact of rising income inequality to the baseline economy found above. 
% So we need to simulate changing sigma_w and then optimise each 
% variable under the new inequality - highlighting the improvement using the optimal tax rate.


sigma_w_inequality_range = sigma_w:0.3:3;           % Creates a range of sigma_w values in a vector - representing increased inequality. 


%%% Initialisation of all vectors to store the results of the 'for' loop below.

% Optimal Variables after the intervention

opt_gammas_intervention = zeros(size(sigma_w_inequality_range));
opt_transfers_intervention = zeros(size(sigma_w_inequality_range));

% Aggregate Variables for the Macroeconomy - Before Intervention

aggregate_income_inequality = zeros(size(sigma_w_inequality_range));
aggregate_consumption_inequality = zeros(size(sigma_w_inequality_range));
aggregate_labor_inequality = zeros(size(sigma_w_inequality_range));

% Aggregate Variables for the Macroeconomy - After Intervention

aggregate_income_intervention = zeros(size(sigma_w_inequality_range));
aggregate_consumption_intervention = zeros(size(sigma_w_inequality_range));
aggregate_labor_intervention = zeros(size(sigma_w_inequality_range));

% Gini Coefficients representing inequality in economy for each sigma_w -
% before intervention 

labordata_gini_coeffs_inequality = zeros(size(sigma_w_inequality_range));
consdata_gini_coeffs_inequality =  zeros(size(sigma_w_inequality_range));
incomedata_gini_coeffs_inequality = zeros(size(sigma_w_inequality_range));

% Gini Coefficients representing inequality in economy for each sigma_w -
% after intervention 
labordata_gini_coeffs_intervention = zeros(size(sigma_w_inequality_range));
consdata_gini_coeffs_intervention = zeros(size(sigma_w_inequality_range));
incomedata_gini_coeffs_intervention = zeros(size(sigma_w_inequality_range));

% Code inside the For loop finds the impact of rising income inequality
% & subsequently optimises each case to find the government response.

for s = 1:length(sigma_w_inequality_range)

    sigma_w = sigma_w_inequality_range(s);          % Current level of sigma_w (income inequality)
    wagedata = exp(normrnd(0, sigma_w, [1,N]));     % Generate log-normal wage data like before associated with this inequality

    % Compute initial weights 

    weights = ones(1,N); % Initialization
    omega = 0.5; % Omega - same as before - threshold for low wage status
    M = 5; % M - same as baseline economy
    for i = 1:N
        if (wagedata(i) <= omega)
            weights(i) = M;
        end
    end
    weights = weights/sum(weights(:)); 

    % Calculate consumption, labor, and income data for each increase in
    % sigma_w

    % Initialisation of vectors
    inequality_consdata = zeros(1, N);
    inequality_labordata = zeros(1, N);
    inequality_incomedata = zeros(1,N);
   
    for i = 1:N
        wage = wagedata(i);
        %inputs here are from the optimsed baseline to show how the economy
        %reacts without intervention. 
        inequality_consdata(i) = cons(wage, lambda, opt_m, opt_gamma, labor(wage,opt_gamma,lambda));  
        inequality_labordata(i) = labor(wage,opt_gamma,lambda);
        inequality_incomedata(i) = wage*inequality_labordata(i);
    end

    % Aggregate calculations
    aggregate_income_inequality(s) = sum(inequality_incomedata);
    aggregate_consumption_inequality(s) = sum(inequality_consdata);
    aggregate_labor_inequality(s) = sum(inequality_labordata);

    % Calculate Gini from the situation where government hasn't intervened
    labordata_gini_coeffs_inequality(s) = gini(ones(1, N), inequality_labordata);
    consdata_gini_coeffs_inequality(s) = gini(ones(1,N),inequality_consdata);
    incomedata_gini_coeffs_inequality(s)= gini(ones(1,N),inequality_incomedata);

    %%% Here, this code is to find the progressivity that maximises the SWF for each sigma_w, to show how the
    %%% government should respond to each level of inequality

    objval = @(gamma) -SWF(wagedata, gamma, N, weights, lambda);
    opt_gamma_intervention = fminbnd(objval, 0, 1);
    opt_gammas_intervention(s) = opt_gamma_intervention;

    %optimal transfers
    opt_m_intervention = transfers(wagedata, opt_gamma_intervention, lambda, N);
    opt_transfers_intervention(s) = opt_m_intervention;

    % Calculate consumption, labor, and income data post-optimal intervention
    post_intervention_consdata = zeros(1, N);
    post_intervention_labordata = zeros(1, N);
    post_intervention_incomedata = zeros(1,N);
    for i = 1:N
        wage = wagedata(i);
        % Inputs now are the sigma_w specific policy choices. Showing the
        % impact of changing government intervention.
        post_intervention_consdata(i) = cons(wage, lambda, opt_m_intervention, opt_gamma_intervention, labor(wage,opt_gamma_intervention,lambda));
        post_intervention_labordata(i) = labor(wage,opt_gamma_intervention,lambda);
        post_intervention_incomedata(i) = wage*post_intervention_labordata(i);
    end

    % Aggregate calculations
    aggregate_labor_intervention(s) = sum(post_intervention_labordata);
    aggregate_consumption_intervention(s) = sum(post_intervention_consdata);
    aggregate_income_intervention(s) = sum(post_intervention_incomedata);

    % Calculate Gini after taxation
    labordata_gini_coeffs_intervention(s) = gini(ones(1, N), post_intervention_labordata);
    consdata_gini_coeffs_intervention(s) = gini(ones(1,N),post_intervention_consdata);
    incomedata_gini_coeffs_intervention(s)= gini(ones(1,N),post_intervention_incomedata);
    
end

%%%% This set of code is to visualise the change in variables before the government choose to intervene 
% and then once they intervene. 
% It is to show how the government should change their policy away from the
% original case in order to respond to the rising inequality and maintain
% maximum social welfare %%% 

% Gini Coefficient Changes Post-taxation vs Pre-taxation for both labour
% and consumption.

figure;
subplot(2,1,1);
plot(sigma_w_inequality_range,labordata_gini_coeffs_inequality, sigma_w_inequality_range,labordata_gini_coeffs_intervention)
xlabel('Income Inequality, \sigma_w');
ylabel('Labor Gini Coefficient');
title('Labor Gini Coefficients Across Different \sigma_w');
legend('Gini Pre-Intervention','Gini Post-Intervention')
legend('boxoff')

subplot(2,1,2);
plot(sigma_w_inequality_range,consdata_gini_coeffs_inequality, sigma_w_inequality_range,consdata_gini_coeffs_intervention)
xlabel('Income Inequality, \sigma_w');
ylabel('Consumption Gini Coefficient');
title('Consumption Gini Coefficients Across Different \sigma_w');


% Here we plot how the government should change their progressivity after inequality has
% increased to continue to maximise the SWF. 

% In the same figure, we show how the government should change their transfers after inequality 
% has increased to continue to maximise the SWF. Computed with a 
% Positive Monotonic Transformation using log to get into a more readable
% scale 

log_opt_transfers = log(opt_transfers_intervention);

figure;
subplot(2,1,1)
plot(sigma_w_inequality_range, log_opt_transfers)
xlabel('Income Inequality, \sigma_w');
ylabel('Optimal Transfers');
title('Optimal Transfers as Inequality, \sigma_w, increases')
subplot(2,1,2)
plot(sigma_w_inequality_range, opt_gammas_intervention)
xlabel('Income Inequality, \sigma_w');
ylabel('Optimal Progressivity');
title('Optimal Progressivity as Inequality, \sigma_w, increases')


% Macroeconomic Effects Visually - the impact on aggregate variables. 

%Pre-tax Aggregate variables - logged to ensure better readability. 

aggregate_labor_inequality_log = log(aggregate_labor_inequality);
aggregate_consumption_inequality_log = log(aggregate_consumption_inequality);

% Post-tax Aggregate Variables -logged

aggregate_consumption_intervention_log = log(aggregate_consumption_intervention);
aggregate_labor_intervention_log = log(aggregate_labor_intervention);

% Plotting aggregate labour and consumption together as progressivity
% decreases - showing how it changes if the optimal taxation policy is
% implemented. Hence, it shows the 'reward' of following the optimal policy
% the model suggests.

figure;
subplot(2,1,1)
plot(opt_gammas_intervention, aggregate_labor_inequality_log, opt_gammas_intervention, aggregate_labor_intervention_log);
set ( gca, 'XDir', 'reverse' ) 
xlabel('Optimal Progressivity')
ylabel('Aggregate Labor Supply')
title('Aggregate Labor Pre vs Post Progressive Taxation');
legend('Labor Pre-Intervention', 'Labor Post-Intervention');
subplot(2,1,2)
plot(opt_gammas_intervention,aggregate_consumption_inequality_log, opt_gammas_intervention,aggregate_consumption_intervention_log);
set ( gca, 'XDir', 'reverse' ) 
xlabel('Optimal Progressivity')
ylabel('Aggregate Consumption')
title('Aggregate Consumption Pre vs Post Progressive Taxation');
legend('Consumprion Pre-Intervention', 'Consumption Post-Intervention');


