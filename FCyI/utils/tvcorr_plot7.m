function tvcorr_plot7(Rho, Date)
% Matrix of time varying correlation plots
%   datenum(Date(3:end,1)), Rho - is matrix of time varying correlations
figure()
P(1)=subplot(6,6,1);
plot(datenum(Date(3:end,1)), Rho(:,1))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(2)=subplot(6,6,7);
plot(datenum(Date(3:end,1)), Rho(:,2))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(3)=subplot(6,6,8);
plot(datenum(Date(3:end,1)), Rho(:,7))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(4)=subplot(6,6,13);
plot(datenum(Date(3:end,1)), Rho(:,3))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(5)=subplot(6,6,14);
plot(datenum(Date(3:end,1)), Rho(:,8))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(6)=subplot(6,6,15);
plot(datenum(Date(3:end,1)), Rho(:,12))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(7)=subplot(6,6,19);
plot(datenum(Date(3:end,1)), Rho(:,4))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(8)=subplot(6,6,20);
plot(datenum(Date(3:end,1)), Rho(:,9))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(9)=subplot(6,6,21);
plot(datenum(Date(3:end,1)), Rho(:,13))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(10)=subplot(6,6,22);
plot(datenum(Date(3:end,1)), Rho(:,16))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(11)=subplot(6,6,25);
plot(datenum(Date(3:end,1)), Rho(:,5))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(12)=subplot(6,6,26);
plot(datenum(Date(3:end,1)), Rho(:,10))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(13)=subplot(6,6,27);
plot(datenum(Date(3:end,1)), Rho(:,14))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(14)=subplot(6,6,28);
plot(datenum(Date(3:end,1)), Rho(:,17))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(15)=subplot(6,6,29);
plot(datenum(Date(3:end,1)), Rho(:,19))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(16)=subplot(6,6,31);
plot(datenum(Date(3:end,1)), Rho(:,6))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(17)=subplot(6,6,32);
plot(datenum(Date(3:end,1)), Rho(:,11))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(18)=subplot(6,6,33);
plot(datenum(Date(3:end,1)), Rho(:,15))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(19)=subplot(6,6,34);
plot(datenum(Date(3:end,1)), Rho(:,18))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(20)=subplot(6,6,35);
plot(datenum(Date(3:end,1)), Rho(:,20))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(21)=subplot(6,6,36);
plot(datenum(Date(3:end,1)), Rho(:,21))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);

h=axes;

set(h,'Visible', 'off');
suptitle('Matrix of Time Varying Correlation Plots');
% linkaxes(P, 'y');

end

