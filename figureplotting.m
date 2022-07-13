isfeas = csvread('iffeas.csv');
ratio = csvread('mod_ratio.csv');

idx_feas = find(isfeas == 1);
idx_infe = find(isfeas == 0);
figure
hold on
grid on
plot(ratio(idx_feas,1),ratio(idx_feas,2),'b*')
plot(ratio(idx_infe,1),ratio(idx_infe,2),'r*')