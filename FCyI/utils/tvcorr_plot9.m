function tvcorr_plot9(Rho, Date)
% Matrix of time varying correlation plots
%   datenum(Date(3:end,1)), Rho - is matrix of time varying correlations
figure()
P(1)=subplot(8,8,1);
plot(datenum(Date(3:end,1)), Rho(:,1))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(2)=subplot(8,8,9);
plot(datenum(Date(3:end,1)), Rho(:,2))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(3)=subplot(8,8,10);
plot(datenum(Date(3:end,1)), Rho(:,9))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(4)=subplot(8,8,17);
plot(datenum(Date(3:end,1)), Rho(:,3))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(5)=subplot(8,8,18);
plot(datenum(Date(3:end,1)), Rho(:,10))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(6)=subplot(8,8,19);
plot(datenum(Date(3:end,1)), Rho(:,16))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(7)=subplot(8,8,25);
plot(datenum(Date(3:end,1)), Rho(:,4))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(8)=subplot(8,8,26);
plot(datenum(Date(3:end,1)), Rho(:,11))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(9)=subplot(8,8,27);
plot(datenum(Date(3:end,1)), Rho(:,17))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(10)=subplot(8,8,28);
plot(datenum(Date(3:end,1)), Rho(:,22))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(11)=subplot(8,8,33);
plot(datenum(Date(3:end,1)), Rho(:,5))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(12)=subplot(8,8,34);
plot(datenum(Date(3:end,1)), Rho(:,12))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(13)=subplot(8,8,35);
plot(datenum(Date(3:end,1)), Rho(:,18))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(14)=subplot(8,8,36);
plot(datenum(Date(3:end,1)), Rho(:,23))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(15)=subplot(8,8,37);
plot(datenum(Date(3:end,1)), Rho(:,27))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(16)=subplot(8,8,41);
plot(datenum(Date(3:end,1)), Rho(:,6))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(17)=subplot(8,8,42);
plot(datenum(Date(3:end,1)), Rho(:,13))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(18)=subplot(8,8,43);
plot(datenum(Date(3:end,1)), Rho(:,19))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(19)=subplot(8,8,44);
plot(datenum(Date(3:end,1)), Rho(:,24))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(20)=subplot(8,8,45);
plot(datenum(Date(3:end,1)), Rho(:,28))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(21)=subplot(8,8,46);
plot(datenum(Date(3:end,1)), Rho(:,31))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(22)=subplot(8,8,49);
plot(datenum(Date(3:end,1)), Rho(:,7))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(23)=subplot(8,8,50);
plot(datenum(Date(3:end,1)), Rho(:,14))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(24)=subplot(8,8,51);
plot(datenum(Date(3:end,1)), Rho(:,20))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(25)=subplot(8,8,52);
plot(datenum(Date(3:end,1)), Rho(:,25))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(26)=subplot(8,8,53);
plot(datenum(Date(3:end,1)), Rho(:,29))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(27)=subplot(8,8,54);
plot(datenum(Date(3:end,1)), Rho(:,32))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(28)=subplot(8,8,55);
plot(datenum(Date(3:end,1)), Rho(:,34))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(29)=subplot(8,8,57);
plot(datenum(Date(3:end,1)), Rho(:,8))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(30)=subplot(8,8,58);
plot(datenum(Date(3:end,1)), Rho(:,15))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(31)=subplot(8,8,59);
plot(datenum(Date(3:end,1)), Rho(:,21))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(32)=subplot(8,8,60);
plot(datenum(Date(3:end,1)), Rho(:,26))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(33)=subplot(8,8,61);
plot(datenum(Date(3:end,1)), Rho(:,30))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(34)=subplot(8,8,62);
plot(datenum(Date(3:end,1)), Rho(:,33))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(35)=subplot(8,8,63);
plot(datenum(Date(3:end,1)), Rho(:,35))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
P(36)=subplot(8,8,64);
plot(datenum(Date(3:end,1)), Rho(:,36))
datetick('x','yyyy');
xlim([min(datenum(Date(3:end,1))) max(datenum(Date(3:end,1)))]); ylim([-1 1]);
h=axes;
set(h,'Visible', 'off');
suptitle('Matrix of Time Varying Correlation Plots');
% linkaxes(P,'y');

end

