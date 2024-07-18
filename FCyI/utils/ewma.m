function [rho Rho] = ewma( S, Median, lambda )
% Calculation of time-varying correlation coefficients by recursively using the exponentially
% weighted moving average (EWMA) method with smoothing factor lambda (RiskMetrics, 1996).
% S is input matrix of used variables
% Median is theoretical median of the time series
% lambda is smoothing factor
T=size(S,1);
K=size(S,2);
% Covariances
for i=1:T
    SS(:,:,i)= (S(i,:)-Median)'*(S(i,:)-Median);
end
 COV(:,:,1) = [ 0.0846    0.0348    0.0112   -0.0103    0.0506    0.0808    0.0650    0.0076    0.0170
    0.0348    0.0846    0.0600   -0.0297   -0.0015    0.0189    0.0044    0.0681    0.0591
    0.0112    0.0600    0.0846   -0.0557    0.0001    0.0044   -0.0089    0.0686    0.0676
   -0.0103   -0.0297   -0.0557    0.0846   -0.0077   -0.0152   -0.0175   -0.0387   -0.0593
    0.0506   -0.0015    0.0001   -0.0077    0.0846    0.0536    0.0475   -0.0082    0.0000
    0.0808    0.0189    0.0044   -0.0152    0.0536    0.0846    0.0694   -0.0045    0.0138
    0.0650    0.0044   -0.0089   -0.0175    0.0475    0.0694    0.0846   -0.0157   -0.0003
    0.0076    0.0681    0.0686   -0.0387   -0.0082   -0.0045   -0.0157    0.0846    0.0668
    0.0170    0.0591    0.0676   -0.0593    0.0000    0.0138   -0.0003    0.0668    0.0846];
   %COV(:,:,1)=zeros(K,K); %cov(S);  
for i=2:T
    COV(:,:,i)= lambda*COV(:,:,i-1) + (1-lambda)*SS(:,:,i);
end

% Variances
%sigma2(:,:,1)=zeros(1,K); %var(S);
sigma2(:,:,1)=[ 0.0846    0.0846    0.0846    0.0846    0.0846    0.0846    0.0846    0.0846    0.0846];
for i=2:T
    for j=1:K
        SS(i, j)= (S(i,j)-Median)^2;
        sigma2(i,j)= lambda*sigma2(i-1,j) + (1-lambda)*SS(i,j);
        sigma=sqrt(sigma2);
    end
end

for i=1:T
    sig_sig(:,:,i)=sigma(i,:)'*sigma(i,:);
end

% Correlations
for i=1:T
    rho(:,:,i)=COV(:,:,i)./sig_sig(:,:,i);
end

% Correlation plots
for h=1:K-1
    for i=3:T
        for j=h+1:K
            C{h}(i-2,j-h)=rho(j,h,i);
        end
    end
end
Rho=[];
for h=1:K-1
    Rho=[Rho C{h}];
end

end

