load params 
% "true" parameters of the model

[exogenous_process, pi] = TAUCHEN(2, p.rho, p.sigma_eps, 1);

T = 160;
rng(0)
rand_numbers = rand(T,1);
exogenous_state = zeros(T,1);
exogenous_state(1) = 1;
cdfs = cumsum(pi,2);
for t = 1:T-1
    exogenous_state(t+1) = find(...
        rand_numbers(t) <= cdfs(exogenous_state(t),:), 1, 'first') ;
end

pol1 = 6.5 + exogenous_process(exogenous_state) + p.sigma_1 * randn(T,1);
pol2 = exp(exogenous_process(exogenous_state)) .^ 1.5 + p.sigma_2 * randn(T,1);

data = cat(3,pol1,pol2);
clearvars -except data
















% p.rho = 0.75;
% p.sigma_eps = 1;
% p.sigma_1 = 1;
% p.sigma_2 = 1;
