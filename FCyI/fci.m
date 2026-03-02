%% Measuring the Financial Cycle
% Authors Hayk Sargsyan & Qnarik Ayvazyan

    % Inputs - Database with transformed data (Gaussian kernel estimate of 
    % the cumulative distribution function) 
    % Outputs - FCI index with complementary graphs
    % Steps - In the settings part you should define the desired inputs
    %       . n - Number of iteration for different weight distributions.
    %       . lambda - Smoothing factor in method (RiskMetrics, 1996).    
    %       . Median - "Theoretical" median.
    %       . Country - Armenia / Czeck.
clear all;
close all;
addpath utils;

%% Settings
n=30000;  
lambda=0.94;
Median=0.5;  
Country={'Armenia'};
%% Data Loading 
S = readtable('Armenia_data.xlsx', 'Sheet', 'data');
Y = readtable('Armenia_data.xlsx', 'Sheet', 'Y');
date = S.Date;
columnNames = S.Properties.VariableNames;

% %% Initial Plot
% figure();
% for i=2:K
%     subplot(3,3,i-1)
%     plot(S{:, 'Date'}, S{:, columnNames{i}});
%     datetick('x','yyyy');
%     % xlim([min(datetime(date)) max(datetime(date))]); ylim([0 1]);
%     title(columnNames{i});
% end

% converting to array
S = S{:,2:end};
Y = Y{:, 2};
T = size(S,1);
K = size(S,2);

%% Correlation Matrix with EWMA (Exponentially Weighted Moving Average)
[rho, Rho]=ewma(S,Median,lambda);

% Simulation of Weights
W = weight_simulation(S, rho, Y, date,  n);
W=[ 0.2	0.2	0.05	0.086	0.08	0.26	0.048	0.046	0.036]; % Weights by paper

%% FCI

E1=bsxfun(@times, W, S);
for i=1:T
    E2(:,:,i)=E1(i,:)*rho(:,:,i);
    FCI(i,1)=E2(:,:,i)*E1(i,:)';
end

EP1=bsxfun(@times, W,S);
for i=1:T
    rho1(:,:,i)=ones(K,K);
end
for i=1:T
    EP2(:,:,i)=EP1(i,:)*rho1(:,:,i);
    FCIP(i,1)=EP2(:,:,i)*EP1(i,:)';
    ccc(i,:)=EP1(i,:).*EP2(:,:,i);
end

c =[FCI-FCIP];
[T]=hpfilter(FCI, 0.5);
M= median(T);

% FCI Plot
figure();
plot(date,T);
datetick('x','yyyy');
xlim([min(date) max(date)]);
title('FCI Index','FontSize',14);

% Smooth FCI and Median
% figure();
% plot(datenum(date),T), hold on;
% plot(datenum(date),M);
% datetick('x','yyyy');
% xlim([min(datenum(date)) max(datenum(date))]);
% title('FCI Index','FontSize',14);

% Contribution Plot
C=[ ccc c];
if strcmp(Country,'Armenia')
    contrplot9(T, C, date);
    legend('Credit spread NFC','Credit spread households','Trade account deficit/GDP','Credit to GDP GAP','Total loans GDP YoY','REP growth rate','Corporate flows','Mortgage flows','Consumer flows','Correlation contribution','FCI')
else
        contrplot9(FCI, C, date);
end
