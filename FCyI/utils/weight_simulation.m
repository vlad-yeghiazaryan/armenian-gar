function W = weight_simulation(S, rho, Y,date, n)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
rng default;
T=size(S,1);
K=size(S,2);
w=sort(rand(K,n),1, 'descend');
s=sum(w);
w=bsxfun(@rdivide,w,s);
% figure()
% plot(w)
% title('Randomly Generated Weights','FontSize',14);

for m=1:n
E1=bsxfun(@times, w(:,m)',S);
for i=1:T
    E2(:,:,i)=E1(i,:)*rho(:,:,i);
    FCI(i,m)=E2(:,:,i)*E1(i,:)';
end
end

% figure()
% plot(datenum(date),FCI); 
% datetick('x','yyyy');
% xlim([min(datenum(date)) max(datenum(date))]);
% title('Pool of FCI Index','FontSize',14)




% RMSE
for m=1:n
    [~,~,R] = regress(Y,FCI(:,m));
    R(isnan(R))=[];
    R2=R.^2;
    RMSE(:,m)=sqrt(mean(R2));
end
[~, I]=min(RMSE);
W=w(:,I)';


end

