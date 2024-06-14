% Competing FCIs - Forecasts from competing models
% To minimize programming error, I treat the competing FCIs as FAVAR models with a known factor (FCI),
% hence I am using the same code as the FAVAR models.
%-----------------------------------------------------------------------------------------
% The model is:
%     _    _     _              _     _    _     _    _
%    | y[t] |   |   I        0   |   | y[t] |   |   0  |
%    |      | = |                | x |      | + |      |
%	 | x[t] |   | L[y,t]  L[f,t] |   | f[t] |   | e[t] |
%     -    -     -              -     -    -     -    -
%	 
%     _    _              _      _
%    | y[t] |            | y[t-1] |   
%    |      | = B[t-1] x |        | + u[t]
%    | f[t] |            | f[t-1] |   
%     -    -              -      -     
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

%-------------------------------USER INPUT--------------------------------------
nlag = 2;              % maximum number of lags
rolling = [20 30];  % sample lengths for rolling forecasts

% Forecasting
nfore = 5;        % Forecast horizon (note: forecasts are iterated)
nsim = 1000;      % Number of times to simulate from the predictive density
t0 = '1999Q4';    % Set initial estimation period
              
%----------------------------------LOAD DATA----------------------------------------
% Load Koop and Korobilis (2012) quarterly data
load xdata.dat;
% load data on inflation, gdp and unemployment
load macro_p.dat;
load macro_ruc.dat;
load macro_routput.dat;
% load data on alternative FCIs (from central banks)
load other_FCIs.dat;
% load the file with the dates of the data (quarters)
load yearlab.mat;
% load the file with the names of the variables
load varnames.mat;
namesXY = ['Inflation' ; 'Unemployment'; 'GDP'; varnames ];

% Convert t0 to numeric value
t0=find(strcmp(yearlab,t0)==1);
t1=find(strcmp(yearlab,'1989Q4')==1);

% Demean and standardize data (needed to extract Principal Components)
xdata = standardize_miss(xdata) + 1e-40 ;
xdata(isnan(xdata)) = 0;
% ydata = standardize(ydata);

% Define X and Y matrices
X = xdata;   % X contains the 'xdata' which are used to extract factors.
Y = [macro_p(:,end), macro_ruc(:,end), macro_routput(:,end)];   % Y contains inflation, gdp and unemployment

% Number of observations and dimension of X and Y
t = size(Y,1); % t time series observations
p = size(Y,2); % and p macro series

% ======================| Form all possible model combinations |======================
other_FCIs = [other_FCIs, zeros(t,1)]; % Augment the vector to incorporate the PCA
KK = size(other_FCIs,2);  % Number of models is number of FCIs + principal component

% =========================| PRIORS |================================
beta_rec = zeros(nlag*2+1,nsim,p,KK);
beta_rol = zeros(nlag*2+1,nsim,p,KK);
sigma_rec = zeros(nsim,p,KK);
sigma_rol = zeros(nsim,p,KK);

% Initialize matrix of forecasts
y_fore_rec = zeros(nfore,p,nsim,KK,nlag);
y_fore_rol = zeros(nfore,p,nsim,KK,nlag*length(rolling));
y_t_other = zeros(nfore,p,KK,t-t0);
Yraw_f = [[macro_p(:,end), macro_ruc(:,end), macro_routput(:,end)] ; NaN(nfore,p)];
PL = zeros(p,KK);

MSFE_rec_selection = zeros(t-(t0-4),p,nfore,KK,nlag);
MSFE_rol_selection = zeros(t-(t0-4),p,nfore,KK,nlag*length(rolling));

MAFE_rec = zeros(t-t0,p,nfore,KK);
MSFE_rec = zeros(t-t0,p,nfore,KK);
FE_draw_sq_rec = zeros(t-t0,p,nfore,nsim,KK);
PL_rec = zeros(t-t0,p,nfore,KK);

MAFE_rol = zeros(t-t0,p,nfore,KK);
MSFE_rol = zeros(t-t0,p,nfore,KK);
FE_draw_sq_rol = zeros(t-t0,p,nfore,nsim,KK);
PL_rol = zeros(t-t0,p,nfore,KK);

%======================= BEGIN KALMAN FILTER ESTIMATION =======================
tic;

for irep = t0-4:t-1
%     if mod(irep,ceil(t./40)) == 0
        disp([num2str(100*((irep-t0)/(t-t0-1))) '% completed'])       
        toc;
%     end
    
    % Standardize data up to time irep     
    X_st = standardize_miss(xdata(1:irep,:));% + 1e-20;
    X_st(isnan(X_st)) = 0;  
    Y = [macro_p(1:irep,irep+1-t0 + (t0-t1)), macro_ruc(1:irep,irep+1-t0 + (t0-t1)), macro_routput(1:irep,irep+1-t0 + (t0-t1))];
    X = X_st; 
    [FPC,L] = extract(X,1);    
    
    other_FCIs(1:irep,end) = FPC;
    
    % Iterate over all models
    y_f_draw = zeros(nfore,p,nsim,KK);
    for nmod = 1:KK
        % Extract Principal Components using data up to time irep        
        F = standardize_miss(other_FCIs(1:irep,nmod));
        F(isnan(F))=0;
        
        for ifore = 1:nfore
            % Estimate Univariate regression with diffuse prior
            for ivar = 1:p % Estimate for each macro variable (prices, unemployment, output)
                ind_rol = 0;
                for nlag = 1:4                    
                    % 1) First calculcate everything recursively (for recursive forecasts)
                    Y_Reg = Y(nlag+ifore+1:end,ivar);
                    Y_lag = mlag2(Y(:,ivar),nlag); F_lag = mlag2(F,nlag);
                    t_temp = size(Y_Reg,1);
                    X_Reg = [ones(t_temp,1) Y_lag(nlag+1:end-ifore,:) F_lag(nlag+1:end-ifore,:)];
                    iXX = inv(X_Reg'*X_Reg);
                    k = size(X_Reg,2);
                       
                    % Obtain OLS quantities
                    beta_OLS = inv(X_Reg'*X_Reg)*(X_Reg'*Y_Reg);
                    sigma_OLS = (Y_Reg - X_Reg*beta_OLS)'*(Y_Reg - X_Reg*beta_OLS)./(t_temp-k);
            
                    % Calculate posterior of diffuse prior
                    for isim = 1:nsim
                        sigma_rec(isim,ivar,nmod) = 1./gamrnd(.5*(t_temp-k),2./((t_temp-k)*sigma_OLS));                      
                        beta_var = sigma_rec(isim,ivar,nmod)*iXX;
                        beta_mean = beta_OLS;
                        beta_rec(1:nlag*2+1,isim,ivar,nmod) = beta_mean + chol(beta_var)'*randn(k,1);
                        y_fore_rec(ifore,ivar,isim,nmod,nlag) = [1 Y_lag(end,:) F_lag(end,:)]*beta_rec(1:nlag*2+1,isim,ivar,nmod) + sqrt(sigma_rec(isim,ivar,nmod))*randn;
                    end
                    for r_s = 1:length(rolling)
                        ind_rol = ind_rol+1;
                        rol_samp = rolling(r_s);
                        % 2) Now do the same using rolling samples of 25 quarters (for rollings forecasts)
                        Y_Reg = Y(end-(rol_samp-1):end,ivar);
                        Y_lag = mlag2(Y(:,ivar),nlag); F_lag = mlag2(F,nlag);
                        t_temp = size(Y_Reg,1);
                        X_Reg = [ones(t_temp,1) Y_lag(end-(rol_samp-1)-ifore:end-ifore,:) F_lag(end-(rol_samp-1)-ifore:end-ifore,:)];
                        iXX = inv(X_Reg'*X_Reg);
                        k = size(X_Reg,2);
                           
                        % Obtain OLS quantities
                        beta_OLS = inv(X_Reg'*X_Reg)*(X_Reg'*Y_Reg);
                        sigma_OLS = (Y_Reg - X_Reg*beta_OLS)'*(Y_Reg - X_Reg*beta_OLS)./(t_temp-k);
                        
                        % Calculate posterior of diffuse prior
                        for isim = 1:nsim                                          
                            sigma_rol(isim,ivar,nmod) = 1./gamrnd(.5*(t_temp-k),2./((t_temp-k)*sigma_OLS));                      
                            beta_var = sigma_rol(isim,ivar,nmod)*iXX;
                            beta_mean = beta_OLS;
                            beta_rol(1:nlag*2+1,isim,ivar,nmod) = beta_mean + chol(beta_var)'*randn(k,1);
                            y_fore_rol(ifore,ivar,isim,nmod,nmod,ind_rol) = [1 Y_lag(end,:) F_lag(end,:)]*beta_rol(1:nlag*2+1,isim,ivar,nmod) + sqrt(sigma_rol(isim,ivar,nmod))*randn;
                        end                      
                    end                    
                end
            end
        end
        if irep>t0-4
            ind_rol = 0;
            for nlag = 1:4
                mean_fore_rec = squeeze(mean(y_fore_rec(:,:,:,nmod,nlag),3));
                MSFE_rec_selection(irep-(t0-4),:,1,nmod,nlag) = ( Yraw_f(irep+1,:) - mean_fore_rec(1,:) ).^2;
                for r_s = 1:length(rolling)
                    ind_rol = ind_rol+1;                       
                    mean_fore_rol = squeeze(mean(y_fore_rol(:,:,:,nmod,ind_rol),3));
                    MSFE_rol_selection(irep-(t0-4),:,1,nmod,ind_rol) = ( Yraw_f(irep+1,:) - mean_fore_rec(1,:) ).^2;
                end
            end
        end
    end
    if irep>t0
        for nmod = 1:KK
            best_rec = zeros(p,1);
            best_rol = zeros(p,1);
            for ivar = 1:p
                [~,be_re] = max(mean(MSFE_rec_selection(irep-(t0-1):irep-(t0-4),ivar,1,nmod,:)));                       
                [~,be_ro] = max(mean(MSFE_rol_selection(irep-(t0-1):irep-(t0-4),ivar,1,nmod,:)));
                best_rec(ivar,1) = be_re;
                best_rol(ivar,1) = be_ro;                
                mean_fore_rec(:,ivar) = mean(y_fore_rec(:,ivar,:,nmod,be_re),3);
                mean_fore_rol(:,ivar) = mean(y_fore_rol(:,ivar,:,nmod,be_ro),3);
            end
            for ii = 1:nfore          
                FE_draw_sq_rec(irep-t0,:,ii,:,nmod) = ( repmat(Yraw_f(irep+ii,:)',1,nsim) - squeeze(y_fore_rec(ii,:,:,nmod)) ).^2;
                FE_draw_sq_rol(irep-t0,:,ii,:,nmod) = ( repmat(Yraw_f(irep+ii,:)',1,nsim) - squeeze(y_fore_rol(ii,:,:,nmod)) ).^2;
                MSFE_rec(irep-t0,:,ii,nmod) = ( Yraw_f(irep+ii,:) - mean_fore_rec(ii,:) ).^2;
                MSFE_rol(irep-t0,:,ii,nmod) = ( Yraw_f(irep+ii,:) - mean_fore_rol(ii,:) ).^2;
                MAFE_rec(irep-t0,:,ii,nmod) = abs( Yraw_f(irep+ii,:) - mean_fore_rec(ii,:) );
                MAFE_rol(irep-t0,:,ii,nmod) = abs( Yraw_f(irep+ii,:) - mean_fore_rol(ii,:) );
                for jj = 1:p
                    PL_rec(irep-t0,jj,ii,nmod) = ksdensity(squeeze(y_fore_rec(ii,jj,:,nmod)),Yraw_f(irep+ii,jj));
                    PL_rol(irep-t0,jj,ii,nmod) = ksdensity(squeeze(y_fore_rol(ii,jj,:,nmod)),Yraw_f(irep+ii,jj));
                end
            end
        end
    end
end
%======================== END KALMAN FILTER ESTIMATION ========================
clc;
toc;

model = 'Diffusion_Index_Forecasting_FULL2';
save(sprintf('%s_%g.mat',model,nlag),'-mat');

