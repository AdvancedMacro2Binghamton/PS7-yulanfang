function [LLH] = model_llh(params, data, N, T)
p.rho = params(1);
p.sigma_eps = params(2);
p.sigma_1 = params(3);
p.sigma_2 = params(4);

T = min(T, length(data));

%%% Model-implied transition equations:
% state transition:
[exo_process, pi] = TAUCHEN(2, p.rho, p.sigma_eps, 1);
% ergodic long-run distribution:
lr_dist = (zeros(size(exo_process')) + 1 / length(exo_process)) * pi ^ 50;

%%% Empirical log-likelihoods by particle filtering
% initialize particles according to S_0
cdfs = cumsum(lr_dist);
rng(0)
init_seeds = rand(1,N);

particles = zeros(T, N);
llhs = zeros(T,1);

particles(1, init_seeds <= cdfs(1)) = 1;
for ii = 2:length(lr_dist)
    particles(1, init_seeds > cdfs(ii-1) & init_seeds <= cdfs(ii)) = ii; 
end
llhs(1) = log( mean( exp( ...
        log( normpdf(data(1,:,1), 6.5 + exo_process(particles(1,:)), p.sigma_1) ) + ...
        log( normpdf(data(1,:,2), exp(exo_process(particles(1,:))) .^ 1.5, p.sigma_2) ) ...
        ) ) );

% predict, filter, update particles and collect the likelihood 
cdfs = cumsum(pi,2);

for t = 2:T
    %%% Prediction:
    shocks = rand(1,N);
    for n = 1:N
        particles(t,n) = find( cdfs(particles(t-1,n),:) >= shocks(n), 1, 'first' );
    end
    
    %%% Filtering:
    llh = log( normpdf(data(t,:,1), 6.5 + exo_process(particles(t,:)), p.sigma_1) ) + ...
        log( normpdf(data(t,:,2), exp(exo_process(particles(t,:))) .^ 1.5, p.sigma_2) );
    lh = exp(llh);
    weights = exp( llh - log( sum(lh) ) );
    % store the log(mean likelihood)
    llhs(t) = log(mean(lh));
    
    %%% Sampling:
    particles(t,:) = datasample(particles(t,:), N, 'Weights', weights);
    
end

LLH = sum(llhs);