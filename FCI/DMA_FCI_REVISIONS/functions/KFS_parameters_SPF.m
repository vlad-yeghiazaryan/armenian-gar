function [beta_t,beta_new,Sb_t,lambda_t,Sl_t,V_t,Q_t] = KFS_parameters_SPF(YX,FPC,YFF,l,nfac,nlag,k,m,p,n,r,rr,t,lambda_0,beta_0,V_0,Q_0)

% Function to estimate time-varying loadings, coefficients, and covariances
% from a TVP-FAVAR, conditional on feeding in an estimate of the factors
% (Principal Components). This function runs the Kalman filter and smoother
% for all time-varying parameters using an adaptive algorithm (EWMA filter
% for the covariances).
%
% Written by Dimitris Korobilis, 2012
% University of Glasgow

% Initialize matrices
lambda_0_prmean = lambda_0.mean;
lambda_0_prvar = lambda_0.var;
beta_0_prmean = beta_0.mean;
beta_0_prvar = beta_0.var;

lambda_pred = zeros(n,rr,t);
lambda_update = zeros(n,rr,t);
beta_pred = zeros(m,t);
beta_update = zeros(m,t);

Rl_t = zeros(rr,rr,n,t);
Sl_t = zeros(rr,rr,n,t);
Rb_t = zeros(m,m,t);
Sb_t = zeros(m,m,t);

x_t_pred = zeros(t,n);
e_t = zeros(n,t);
lambda_t = zeros(n,rr,t);
beta_t = zeros(k,k,t);
Gf_t = zeros(r,r,t);
Q_t = zeros(r,r,t);
V_t = zeros(n+p,n+p,t);

% Decay and forgetting factors
l_1 = l(1); l_2 = l(2); l_3 = l(3); l_4 = l(4);

% Define lags of the factors to be used in the state (VAR) equation         
yy = FPC(nlag+1:t,:);      
xx = mlag2(FPC,nlag); xx = xx(nlag+1:t,:);
templag = mlag2(FPC,nlag); templag = templag(nlag+1:t,:);      
[Flagtemp,m] = create_RHS_NI(templag,r,nlag,t);  
Flag = [zeros(k,m); Flagtemp];

% ======================| 1. KALMAN FILTER
for irep = 1:t  
    % -----| Update the state covariances
    % 1. Get the variance of the factor
    % Update Q[t]
    if irep==1
        Gf_t(:,:,irep) = Q_0;
        Q_t(:,:,irep) = Q_0;
    elseif irep>1
        if irep<=nlag+1
            Gf_t(:,:,irep) = 0.1*(FPC(irep,:)'*FPC(irep,:));
        else
            Gf_t(:,:,irep) = (yy(irep-nlag,:)-xx(irep-nlag,:)*B(1:r,1:k)')'*(yy(irep-nlag,:)-xx(irep-nlag,:)*B(1:r,1:k)');
        end
        if l_2 == 1
            if irep < 10
                Q_t(:,:,irep) = Q_0;
            else
                Q_t(:,:,irep) = squeeze(mean(Gf_t(1:r,1:r,:),3));
            end
        else
            Q_t(:,:,irep) = l_2*Q_t(:,:,irep-1) + (1-l_2)*Gf_t(1:r,1:r,irep);
        end        
    end
    
    % =======| Kalman predict steps
    %  -for lambda
    if irep==1
        lambda_pred(:,:,irep) = lambda_0_prmean;
        for i = 1:n
            Rl_t(:,:,i,irep) = lambda_0_prvar;
        end
    elseif irep>1
        lambda_pred(:,:,irep) = lambda_update(:,:,irep-1);
        Rl_t(:,:,:,irep) = (1./l_3)*Sl_t(:,:,:,irep-1);
    end
    % -for beta
    if irep<=nlag+1
        beta_pred(:,irep) = beta_0_prmean;
        beta_update(:,irep) = beta_pred(:,irep);
        Rb_t(:,:,irep) = beta_0_prvar;
    elseif irep>nlag+1
        beta_pred(:,irep) = beta_update(:,irep-1);
        Rb_t(:,:,irep) = (1./l_4)*Sb_t(:,:,irep-1);
    end
    
    % One step ahead prediction based on PC factor
    x_t_pred(irep,:) = lambda_pred(:,:,irep)*YFF(irep,:)';
    % Prediction error
    e_t(:,irep) = YX(irep,p+1:end)' - x_t_pred(irep,:)';
    
    % 3. Get the measurement error variance
    if l_1 == 1
        A_t = e_t(:,1:irep)*e_t(:,1:irep)';
        if irep < 10
            V_t(:,:,irep) = diag(diag(V_0));
        else
            V_t(p+1:end,p+1:end,irep) = (1./irep)*diag(diag(A_t)) + 1e-10*eye(n);
        end        
    else
        A_t = e_t(:,irep)*e_t(:,irep)';
        if irep==1      
            V_t(:,:,irep) = diag(diag(V_0));
        else
            V_t(p+1:end,p+1:end,irep) = l_1*V_t(p+1:end,p+1:end,irep-1) + (1-l_1)*diag(diag(A_t));
        end
    end

    % =======| Kalman update steps
    % -for lambda
    for i = 1:n
        % 1/ Update loadings conditional on Principal Components estimates       
        Rx = Rl_t(1:rr,1:rr,i,irep)*YFF(irep,1:rr)';
        KV_l = V_t(i+p,i+p,irep) + YFF(irep,1:rr)*Rx;
        KG = Rx/KV_l;
        lambda_update(i,1:rr,irep) = lambda_pred(i,1:rr,irep) + (KG*(YX(irep,i+p)'-lambda_pred(i,1:rr,irep)*YFF(irep,1:rr)'))';
        Sl_t(1:rr,1:rr,i,irep) = Rl_t(1:rr,1:rr,i,irep) - KG*(YFF(irep,1:rr)*Rl_t(1:rr,1:rr,i,irep));        
    end
    % -for beta
    if irep>=nlag+1
        % 2/ Update VAR coefficients conditional on Principal Componets estimates
        Rx = Rb_t(:,:,irep)*Flag((irep-1)*r+1:irep*r,:)';
        KV_b = Q_t(:,:,irep) + Flag((irep-1)*r+1:irep*r,:)*Rx;
        KG = Rx/KV_b;
        beta_update(:,irep) = beta_pred(:,irep) + (KG*(FPC(irep,:)'-Flag((irep-1)*r+1:irep*r,:)*beta_pred(:,irep)));
        Sb_t(:,:,irep) = Rb_t(:,:,irep) - KG*(Flag((irep-1)*r+1:irep*r,:)*Rb_t(:,:,irep));
    end    
    
    % Assign coefficients
    bb = beta_update(:,irep);
    splace = 0; biga = 0;
    for ii = 1:nlag                                          
        for iii = 1:r           
            biga(iii,(ii-1)*r+1:ii*r) = bb(splace+1:splace+r,1)';
            splace = splace + r;
        end        
    end
    B = [biga ; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
    %B = [reshape(beta_update(:,irep),r,r*nlag) ; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
    lambda_t(:,:,irep) = lambda_update(:,:,irep);
    if max(abs(eig(B)))<0.9999
         beta_t(:,:,irep) = B;
    else
        beta_t(:,:,irep) = beta_t(:,:,irep-1);
        beta_update(:,irep) = 0.95*beta_update(:,irep-1);
    end
end
    
% ======================| 2. KALMAN SMOOTHER
lambda_new = 0*lambda_update;   beta_new = 0*beta_update;
lambda_new(:,:,t) = lambda_update(:,:,t);  beta_new(:,t) = beta_update(:,t);
Q_t_new = 0*Q_t; Q_t_new(:,:,t) = Q_t(:,:,t);
V_t_new = 0*V_t; V_t_new(:,:,t) = V_t(:,:,t);
for irep = t-1:-1:1
    % 1\ Smooth lambda
    lambda_new(1:rr,:,irep) = lambda_update(1:rr,:,irep);
    for i = 1:n
        Ul_t = Sl_t(1:rr,1:rr,i,irep)/Rl_t(1:rr,1:rr,i,irep+1);
        lambda_new(i,1:rr,irep) = lambda_update(i,1:rr,irep) + (lambda_new(i,1:rr,irep+1) - lambda_pred(i,1:rr,irep+1))*Ul_t';   
    end
    % 2\ Smooth beta    
    if sum(sum(Rb_t(:,:,irep+1))) == 0
        beta_new(:,irep) = beta_update(:,irep);
    else
        Ub_t = Sb_t(:,:,irep)/Rb_t(:,:,irep+1);
        beta_new(:,irep) = beta_update(:,irep) + Ub_t*(beta_new(:,irep+1) - beta_pred(:,irep+1));
    end
    % 3\ Smooth Q_t    
    Q_t_new(:,:,irep) = 0.9*Q_t(:,:,irep) + 0.1*Q_t_new(:,:,irep+1);
    % 4\ Smooth V_t
    V_t_new(p+1:end,p+1:end,irep) = 0.9*V_t(p+1:end,p+1:end,irep) + 0.1*V_t_new(p+1:end,p+1:end,irep+1);       
end

% Assign coefficients 
for irep = 1:t
    bb = beta_new(:,irep);
    splace = 0; biga = 0;
    for ii = 1:nlag                                          
        for iii = 1:r           
            biga(iii,(ii-1)*r+1:ii*r) = bb(splace+1:splace+r,1)';
            splace = splace + r;
        end        
    end
    B = [biga ; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
    lambda_t(:,:,irep) = lambda_new(:,:,irep);
    beta_t(:,:,irep) = B;
end
