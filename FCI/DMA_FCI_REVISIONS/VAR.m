% VAR - Time-varying parameters VAR using an adaptive Kalman filter with EWMA filter covariance estimation 
% SINGLE MODEL CASE: NONINFORMATIVE PRIOR
%-----------------------------------------------------------------------------------------
% The model is:
%    
%   y[t]  = B[t-1] x  y[t-1] + u[t]  
%
% where L[t] = (L[y,t] ; L[f,t]) and B[t] are coefficients, f[t] are factors, e[t]~N(0,V[t])
% and u[t]~N(0,Q[t]), and
% 
%   L[t] = L[t-1] + v[t]
%   B[t] = B[t-1] + n[t]
%
% with v[t]~N(0,H[t]), n[t]~N(0,W[t])
%
% All covariances follow EWMA models of the form:
%
%  V[t] = l_1 V[t-1] + (1 - l_1) e[t-1]e[t-1]'
%  Q[t] = l_2 Q[t-1] + (1 - l_2) u[t-1]u[t-1]'
%
% with l_1, l_2, l_3 and l_4 being the decay/forgetting factors (see paper for details).
%-----------------------------------------------------------------------------------------
%  - This code does NOT estimate an FCI (nfac=0, simple VAR case without FCI)
%  - This code uses NONINFORMATIVE PRIORS AND INITIAL CONDITIONS
%-----------------------------------------------------------------------------------------
% Written by Dimitris Korobilis
% University of Glasgow
% This version: 08 July, 2013
%-----------------------------------------------------------------------------------------

clear all;
close all;
clc;

% Add path of data and functions
addpath('data');
addpath('functions');
for l_1 = [0.92, 0.94, 0.96, 1.00]
    l_2 = l_1;
    for l_3 = [0.98, 0.99, 1.00]
        l_4 = l_3;
%-------------------------------USER INPUT--------------------------------------
% Model specification
nfac = 0;         % number of factors
nlag = 4;         % number of lags of factors

% Control the amount of variation in the measurement and error variances
% l_1 = 0.96;       % Decay factor for measurement error variance
% l_2 = 0.96;       % Decay factor for factor error variance
% l_3 = 0.99;       % Forgetting factor for loadings error variance
% l_4 = 0.99;       % Forgetting factor for VAR coefficients error variance

% Select if SPF[t] should be included in the measurement equation (if it is
% NOT included, then the coefficient/loading L[SPF,t] is zero for all periods)
y_true = 1;       % 1: Purge FCI; 0: Do not purge FCI                               
                  
% Forecasting
nfore = 5;        % Forecast horizon (note: forecasts are iterated)
nsim = 2000;      % Number of times to simulate from the predictive density
t0 = '1989Q4';    % Set last observation of initial estimation period  
              
%----------------------------------LOAD DATA----------------------------------------
% Load Koop and Korobilis (2012) quarterly data
% load data used to extract factors
load xdata.dat;
% load data on inflation, gdp and unemployment
load macro_p.dat;
load macro_ruc.dat;
load macro_routput.dat;
% load transformation codes (see file transx.m)
load tcode.dat;
% load the file with the dates of the data (quarters)
load yearlab.mat;
% load the file with the names of the variables
load varnames.mat;
namesXY = ['Inflation' ; 'Unemployment'; 'GDP'; varnames ];

% Convert t0 to numeric value
t0=find(strcmp(yearlab,t0)==1);

% Demean and standardize data (needed to extract Principal Components)
xdata = standardize_miss(xdata) + 1e-40 ;
xdata(isnan(xdata)) = 0;
% ydata = standardize(ydata);

% Define X and Y matrices
X = xdata;   % X contains the 'xdata' which are used to extract factors.
Y = [macro_p(:,end), macro_ruc(:,end), macro_routput(:,end)];   % Y contains inflation, gdp and unemployment

% Number of observations and dimension of X and Y
t = size(Y,1); % t time series observations
n = size(X,2); % n series from which we extract factors
p = size(Y,2); % and p macro series
r = nfac + p;  % number of factors and macro series
q = n + p;     % number of observed and macro series

% Set dimensions of useful quantities
m = nlag*(r^2);  % number of VAR parameters
k = nlag*r;         % number of sampled factors

% =========================| PRIORS |================================
% Initial condition on the factors
factor_0.mean = zeros(k,1);
factor_0.var = 4*eye(k);
% Initial condition on lambda_t
lambda_0.mean = zeros(q,r);
lambda_0.var = 4*eye(r);
% Initial condition on beta_t
[b_prior,Vb_prior] = Minn_prior_KOOP(.1,r,nlag,m); % Obtain a Minnesota-type prior
beta_0.mean = b_prior;
beta_0.var = Vb_prior;
% Initial condition on the covariance matrices
V_0 = 1*eye(q); V_0(1:p,1:p) = 0;
Q_0 = 1*eye(r);

% Put all decay/forgetting factors together in a vector
l = [l_1; l_2; l_3; l_4];

% Initialize matrix of forecasts
y_fore = zeros(nfore,p);
Yraw_f = [[macro_p(:,end), macro_ruc(:,end), macro_routput(:,end)] ; NaN(nfore,p)];

MAFE_final = zeros(t-t0,p,nfore);
MSFE_final = zeros(t-t0,p,nfore);
FE_draw_sq = zeros(t-t0,p,nfore,nsim);
PL_final = zeros(t-t0,p,nfore);

%======================= BEGIN KALMAN FILTER ESTIMATION =======================
tic;
for irep = t0:t-1
    %if mod(irep,ceil(t./40)) == 0
        disp([num2str(100*((irep-t0)/(t-t0-1))) '% completed'])       
        toc;   
    %end
    % Standardize data up to time irep
    if irep<t0
        X_st = standardize_miss(xdata(1:t0,:));% + 1e-20;   
        X_st(isnan(X_st)) = 0;               
        %[Y,Ymeans,Ystds] = standardize2(ydata(1:t0,:));
        Y = [macro_p(1:t0,1), macro_ruc(1:t0,1), macro_routput(1:t0,1)];
    elseif irep>=t0      
        X_st = standardize_miss(xdata(1:irep,:));% + 1e-20;
        X_st(isnan(X_st)) = 0;
        %[Y,Ymeans,Ystds] = standardize2(ydata(1:irep,:));   
        Y = [macro_p(1:irep,irep+1-t0), macro_ruc(1:irep,irep+1-t0), macro_routput(1:irep,irep+1-t0)];
    end
        
    % Extract Principal Components using data up to time irep
    X = X_st;
    [F,L] = extract(X,nfac);
    F(isnan(F))=0;   
    YX = [Y,X];
    YF = [Y,F];
    [L_OLS,B_OLS,beta_OLS,SIGMA_OLS,Q_OLS] = ols_pc_dfm(YX,YF,L,y_true,n,p,r,nfac,nlag);
    
    % Estimate the FCI using the method in Koop and Korobilis (2013):
    % ====| STEP 1: Update Parameters Conditional on PC
    [beta_t,beta_new,Sb_t,lambda_t,Sl_t,V_t,Q_t] = KFS_parameters(YX,YF,l,nfac,nlag,y_true,k,m,p,q,r,irep,lambda_0,beta_0,V_0,Q_0);
    
    % ====| STEP 2: Update Factors Conditional on TV-Parameters
    [factor_new,Sf_t_new] = KFS_factors(YX,lambda_t,beta_t,V_t,Q_t,nlag,k,r,q,irep,factor_0);
    %=========
    if irep>t0 && irep<t   
        YFn = Y;%[Y,factor_new(r,:)'];
        YY = YFn(nlag+1:end,:); XX = mlag2(YFn,nlag); XX = XX(nlag+1:end,:);
        factors = [YFn(end,:) XX(end,1:end-r)]';
        y_f_draw = zeros(nfore,p,nsim);
        for isim = 1:nsim
            VAR_FOR = zeros(k,k);
            for ifore = 1:nfore
                beta_draw = beta_new(:,end) + chol(Sb_t(:,:,end))'*randn(m,1);                  
                splace = 0; biga = 0;
                for ii = 1:nlag                                          
                    for iii = 1:r           
                        biga(iii,(ii-1)*r+1:ii*r) = beta_draw(splace+1:splace+r,1)';
                        splace = splace + r;
                    end
                end
                B = [biga ; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
                Q = [Q_t(:,:,end) zeros(r,r*(nlag-1)); zeros(r*(nlag-1),r*nlag)];
                MEAN_FOR = (B^ifore)*factors;
                VAR_FOR = VAR_FOR + (B^(ifore-1))*Q*(B^(ifore-1))';
                y_f_draw(ifore,:,isim) = MEAN_FOR(1:p,:) + chol(VAR_FOR(1:p,1:p))'*randn(p,1);
            end
        end
    end
    
    if irep>t0
        mean_fore = mean(y_f_draw,3);
        for ii = 1:nfore
            FE_draw_sq(irep-t0,:,ii,:) = ( repmat(Yraw_f(irep+ii,:)',1,nsim) - squeeze(y_f_draw(ii,:,:)) ).^2;
            MSFE_final(irep-t0,:,ii) = ( Yraw_f(irep+ii,:) - mean_fore(ii,:) ).^2;           
            MAFE_final(irep-t0,:,ii) = abs( Yraw_f(irep+ii,:) - mean_fore(ii,:) );
            for jj = 1:p
                PL_final(irep-t0,jj,ii) = ksdensity(squeeze(y_f_draw(ii,jj,:)),Yraw_f(irep+ii,jj));
            end
        end
    end
end
%======================== END KALMAN FILTER ESTIMATION ========================
clc;
toc;

model = 'SINGLE_VAR';
save(sprintf('%s_%g_%g_%g_%g_%g.mat',model,nlag,l_1,l_2,l_3,l_4),'-mat');

    end
end