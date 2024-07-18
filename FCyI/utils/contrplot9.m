function contrplot9(FCI, C, Date )
% Plots contribution of different subindexes of FCI
%   FCI FCI vector 
%   C matrix of sub indexes + contribution of correlations (n*10)
%   Date date vector
figure();
bar(datenum(Date),C(:,1)+C(:,2)+C(:,3)+C(:,4)+C(:,5)+C(:,6)+C(:,7)+C(:,8)+C(:,9),'FaceColor',[.9 .7 .14],'EdgeColor',[.9 .7 .14],'LineWidth',2); hold on
bar(datenum(Date),C(:,1)+C(:,2)+C(:,3)+C(:,4)+C(:,5)+C(:,6)+C(:,7)+C(:,8),'FaceColor',[.9 .9 .14],'EdgeColor',[.9 .9 .14],'LineWidth',2); hold on
bar(datenum(Date),C(:,1)+C(:,2)+C(:,3)+C(:,4)+C(:,5)+C(:,6)+C(:,7),'FaceColor',[.9 .01 .14],'EdgeColor',[.9 .01 .14],'LineWidth',2); hold on
bar(datenum(Date),C(:,1)+C(:,2)+C(:,3)+C(:,4)+C(:,5)+C(:,6),'FaceColor',[.25 .55 .79],'EdgeColor',[.25 .55 .79],'LineWidth',2); hold on
bar(datenum(Date),C(:,1)+C(:,2)+C(:,3)+C(:,4)+C(:,5),'FaceColor',[.2 .7 .3],'EdgeColor',[.2 .7 .3],'LineWidth',2); hold on
bar(datenum(Date),C(:,1)+C(:,2)+C(:,3)+C(:,4),'FaceColor',[.5 .2 .08],'EdgeColor',[.5 .2 .08],'LineWidth',2); hold on
bar(datenum(Date),C(:,1)+C(:,2)+C(:,3),'FaceColor',[0.8 .2 0.1],'EdgeColor',[0.8 .2 0.1],'LineWidth',2); hold on
bar(datenum(Date),C(:,1)+C(:,2),'FaceColor',[.9 .8 .14],'EdgeColor',[.9 .8 .14],'LineWidth',2); hold on
bar(datenum(Date),C(:,1),'FaceColor',[.8 .03 .14],'EdgeColor',[.8 .03 .14],'LineWidth',2); hold on
bar(datenum(Date),C(:,10),'FaceColor',[0.7 0.7 0.7],'EdgeColor',[0.7 0.7 0.7],'LineWidth',2); hold on
plot(datenum(Date),FCI,'k','LineWidth',3); hold off
datetick('x','yyyy');
xlim([min(datenum(Date)) max(datenum(Date))]);
legend({'9', '8', '7', '6', '5', '4', '3', '2', '1', 'Contr. of Corr', 'FCI'}, 'Location','South', 'Orientation', 'horizontal' );
title('The FCI and its Decomposition','FontSize',14)
end

