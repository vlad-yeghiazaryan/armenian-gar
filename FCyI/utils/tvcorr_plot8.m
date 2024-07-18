function tvcorr_plot8(Rho, Date)
% Matrix of time varying correlation plots
%   datenum(Date(3:end,1)), Rho - is matrix of time varying correlations
figure()
P(1)=subplot(7,7,1);
plot(datenum(Date(3:end,1)), Rho(:,1))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(2)=subplot(7,7,8);
plot(datenum(Date(3:end,1)), Rho(:,2))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(3)=subplot(7,7,9);
plot(datenum(Date(3:end,1)), Rho(:,8))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(4)=subplot(7,7,15);
plot(datenum(Date(3:end,1)), Rho(:,3))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(5)=subplot(7,7,16);
plot(datenum(Date(3:end,1)), Rho(:,9))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(6)=subplot(7,7,17);
plot(datenum(Date(3:end,1)), Rho(:,14))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(7)=subplot(7,7,22);
plot(datenum(Date(3:end,1)), Rho(:,4))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(8)=subplot(7,7,23);
plot(datenum(Date(3:end,1)), Rho(:,10))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(9)=subplot(7,7,24);
plot(datenum(Date(3:end,1)), Rho(:,15))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(10)=subplot(7,7,25);
plot(datenum(Date(3:end,1)), Rho(:,19))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(11)=subplot(7,7,29);
plot(datenum(Date(3:end,1)), Rho(:,5))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(12)=subplot(7,7,30);
plot(datenum(Date(3:end,1)), Rho(:,11))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(13)=subplot(7,7,31);
plot(datenum(Date(3:end,1)), Rho(:,16))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(14)=subplot(7,7,32);
plot(datenum(Date(3:end,1)), Rho(:,20))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(15)=subplot(7,7,33);
plot(datenum(Date(3:end,1)), Rho(:,23))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(16)=subplot(7,7,36);
plot(datenum(Date(3:end,1)), Rho(:,6))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(17)=subplot(7,7,37);
plot(datenum(Date(3:end,1)), Rho(:,12))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(18)=subplot(7,7,38);
plot(datenum(Date(3:end,1)), Rho(:,17))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(19)=subplot(7,7,39);
plot(datenum(Date(3:end,1)), Rho(:,21))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(20)=subplot(7,7,40);
plot(datenum(Date(3:end,1)), Rho(:,24))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(21)=subplot(7,7,41);
plot(datenum(Date(3:end,1)), Rho(:,26))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(22)=subplot(7,7,43);
plot(datenum(Date(3:end,1)), Rho(:,7))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(23)=subplot(7,7,44);
plot(datenum(Date(3:end,1)), Rho(:,13))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(24)=subplot(7,7,45);
plot(datenum(Date(3:end,1)), Rho(:,18))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(25)=subplot(7,7,46);
plot(datenum(Date(3:end,1)), Rho(:,22))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(26)=subplot(7,7,47);
plot(datenum(Date(3:end,1)), Rho(:,25))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(27)=subplot(7,7,48);
plot(datenum(Date(3:end,1)), Rho(:,27))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(28)=subplot(7,7,49);
plot(datenum(Date(3:end,1)), Rho(:,28))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);

h=axes;
set(h,'Visible', 'off');
suptitle('Matrix of Time Varying Correlation Plots');
% linkaxes(P,'y');

end

