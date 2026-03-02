function [factor_new,Sf_t_new] = KFS_factors_SPF(YX,Y_SPF,lambda_t,beta_t,V_t,Q_t,nlag,k,r,p,q,t,factor_0)

% Initialize matrices
factor_0_prmean = factor_0.mean;
factor_0_prvar = factor_0.var;

factor_pred = zeros(k,t);
factor_update = zeros(k,t);

lambda_full_t = zeros(q,r,t);
for j=1:t
    lambda_full_t(1:r,1:r,j) = eye(r);
    lambda_full_t(r:q,r,j) = lambda_t(:,end,j);
end

Rf_t = zeros(k,k,t);
Sf_t = zeros(k,k,t);

x_t_predf = zeros(t,q);
ef_t = zeros(q,t);

% ======================| 1. KALMAN FILTER
for irep = 1:t
    % ==============|Update factors conditional on (tvp) coefficients|======    
    % =======| Kalman predict step for f
    if irep==1
        factor_pred(:,irep) = factor_0_prmean;         
        Rf_t(:,:,irep) = factor_0_prvar;
    elseif irep>1
        factor_pred(:,irep) = beta_t(:,:,irep-1)*factor_update(:,irep-1);
        Rf_t(:,:,irep) = beta_t(:,:,irep-1)*Sf_t(:,:,irep-1)*beta_t(:,:,irep-1)' + [Q_t(:,:,irep) zeros(r,r*(nlag-1)); zeros(r*(nlag-1),r*nlag)];
    end
    
    % One step ahead prediction based on Kalman factor
    x_t_predf(irep,:) = lambda_full_t(:,:,irep)*factor_pred(1:r,irep) + [lambda_t(:,1:size(Y_SPF,2),irep)*Y_SPF(irep,:)' ; zeros(p,1)];
    % Prediction error
    ef_t(:,irep) = YX(irep,:)' - x_t_predf(irep,:)';
    
    % =======| Kalman update step for f
    % 3/ Update the factors conditional on the estimate of lambda_full_t and beta_t
    Rf_l = Rf_t(1:r,1:r,irep)*lambda_full_t(:,:,irep)';
    KV_f = V_t(:,:,irep) + lambda_full_t(:,:,irep)*Rf_l;
    KG = Rf_l/KV_f;
    factor_update(1:r,irep) = factor_pred(1:r,irep) + KG*ef_t(:,irep);
    Sf_t(1:r,1:r,irep) = Rf_t(1:r,1:r,irep) - KG*(lambda_full_t(:,:,irep)*Rf_t(1:r,1:r,irep));  
end

% ======================| 2. KALMAN SMOOTHER
% Rauch–Tung–Striebel fixed-interval smoother for the factors  
factor_new = 0*factor_update;          Sf_t_new = 0*Sf_t;
factor_new(:,t) = factor_update(:,t);  Sf_t_new(:,:,t) = Sf_t(:,:,t);
for irep = t-1:-1:1
    Z_t = (Sf_t(:,:,irep)*beta_t(:,:,irep)');
    U_t = squeeze(Z_t(1:r,1:r)/Rf_t(1:r,1:r,irep+1));   
    factor_new(1:r,irep) = factor_update(1:r,irep) + U_t*(factor_new(1:r,irep+1) - factor_pred(1:r,irep+1));
    Sf_t_new(1:r,1:r,irep) = Sf_t(1:r,1:r,irep) + U_t*(Sf_t(1:r,1:r,irep+1) - Rf_t(1:r,1:r,irep+1))*U_t'; 
end
