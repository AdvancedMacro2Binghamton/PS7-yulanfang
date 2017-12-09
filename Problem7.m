clear all; 
close all; 
clc;
load('data.mat');
N = 500; % number of particles
T = 400; % length of time series

% priors:
prior.rho1 = @(x) unifpdf(x,-1,1);
prior.rho2 = @(x) unifpdf(x,-1,1);
prior.phi1 = @(x) unifpdf(x,-1,1);
prior.phi2 = @(x) unifpdf(x,-1,1);
prior.beta = @(x) unifpdf(x,4,7);
prior.sigma = @(x) lognpdf(x,0,1);
prior.sigma_A = @(x) lognpdf(x,0,1);
prior.sigma_B = @(x) lognpdf(x,0,1);
prior.all = @(p) log(prior.rho1(p(1))) + log(prior.rho2(p(2)))+...
    log(prior.phi1(p(3))) + log(prior.phi2(p(4))) + ...
    log(prior.beta(p(5))) + log(prior.sigma(p(6))) + ...
    log(prior.sigma_A(p(7))) + log(prior.sigma_B(p(8)));

% proposals according to random walk with parameter sd's:
prop_sig.rho1 = 0.05;
prop_sig.rho2 = 0.05;
prop_sig.phi1 = 0.05;
prop_sig.phi2 = 0.05;
prop_sig.beta = 0.05;
prop_sig.sigma = 0.05;
prop_sig.sigma_A = 0.05;
prop_sig.sigma_B = 0.05;
prop_sig.all = [prop_sig.rho1 prop_sig.rho2 prop_sig.phi1 prop_sig.phi2 ...
    prop_sig.beta prop_sig.sigma prop_sig.sigma_A prop_sig.sigma_B];

% initial values for parameters
init_params = [0.5 0.5 0.5 -0.5 5 1 0.3 1];

% length of sample
M = 5000;
acc = zeros(M,1);

llhs = zeros(M,1);
parameters = zeros(M,8);
parameters(1,:) = init_params;

% evaluate model with initial parameters
log_prior = prior.all(parameters(1,:));
llh = model_llh(parameters(1,:), data, N, T);
llhs(1) = log_prior + llh;

% sample
rng(0)
proposal_chance = log(rand(M,1));
prop_step = randn(M,8);
tic;
for m = 2:M
    % proposal draw:
    prop_param = parameters(m-1,:) + prop_step(m,:) .* prop_sig.all;
    
    % evaluate prior and model with proposal parameters:
    prop_prior = prior.all(prop_param);
    if prop_prior > -Inf % theoretically admissible proposal
        prop_llh = model_llh(prop_param, data, N, T);
        llhs(m) = prop_prior + prop_llh;
        if llhs(m) - llhs(m-1) > proposal_chance(m)
            accept = 1;
        else
            accept = 0;
        end
    else % reject proposal since disallowed by prior
        accept = 0;
    end
  
    if accept
        parameters(m,:) = prop_param;
        acc(m) = 1;
    else
        parameters(m,:) = parameters(m-1,:);
        llhs(m) = llhs(m-1);
    end
    
    waitbar(m/M)
end
toc

function [LLH] = model_llh(params, data, N, T)
p.rho1 = params(1);
p.rho2 = params(2);
p.phi1 = params(3);
p.phi2 = params(4);
p.beta = params(5);
p.sigma = params(6);
p.sigma_A = params(7);
p.sigma_B = params(8);

T = min(T, length(data));

rng(0);
t=5000;
x=zeros(t+3,1);
epsilon=p.sigma * randn(t+3,1);

for t = 3:t+3
    x(t) = p.rho1*x(t-1) + p.rho2*x(t-2) + p.phi1*epsilon(t-1) + ...
        p.phi2*epsilon(t-2) + epsilon(t);
end

particle = zeros(T, N , 6);
llhs = zeros(T,1);
init_sample = randsample(t,N);
particle(1,:,1)=x(init_sample+2);
particle(1,:,2)=x(init_sample+1);
particle(1,:,3)=x(init_sample);
particle(1,:,4)=epsilon(init_sample+2);
particle(1,:,5)=epsilon(init_sample+1);
particle(1,:,6)=epsilon(init_sample);

llhs(1) = log( mean( exp( ...
        log( normpdf(log(data(1,1)), particle(1,:,1), p.sigma_A) ) + ...
        log( normpdf(data(1,2), p.beta*particle(1,:,1).^2 , p.sigma_B) )...
        ) ) );

% predict, filter, update particles and collect the likelihood 
    %%% Prediction:
for t = 2:T
    particle(t,:,1) = p.rho1*particle(t-1,:,2) + ...
        p.rho2*particle(t-1,:,3) + p.phi1*particle(t-1,:,5) + ...
        p.phi2*particle(t-1,:,6) + p.sigma*randn(1,N);
    particle(t,:,2) = particle(t-1,:,1);
    particle(t,:,3) = particle(t-1,:,2);
    particle(t,:,4) = p.sigma * randn(1,N);
    particle(t,:,5) = particle(t-1,:,6);
    particle(t,:,6) = particle(t-1,:,5);
  
    %%% Filtering:
    llh = log( normpdf(log(data(t,1)), particle(t,:,1), p.sigma_A) ) + ...
        log( normpdf(data(t,2), p.beta*particle(t,:,1).^2 , p.sigma_B) );
    lh = exp(llh);
    
    weight = exp( llh - log( sum(lh) ) );
    if sum(lh)==0
        weight(:) = 1 / length(weight);
    end
    % store the log(mean likelihood)
    llhs(t) = log(mean(lh));
    
    %%% Sampling:
    particle(t,:,1) = datasample(particle(t,:,1), N, 'Weights', weight);
end

LLH = sum(llhs);
end
