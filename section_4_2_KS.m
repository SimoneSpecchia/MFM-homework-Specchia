clear 
close all
grey = [0.4 0.4 0.4];
addpath('..\00_utils')
set_plot_style

%% =========== LOAD DATA AND COMPUTE DERIVATIVES ==========================
load('kuramoto_sivashinsky_big.mat');

ux = zeros(size(usave));
uxx = zeros(size(usave));
uxxx = zeros(size(usave));
uxxxx = zeros(size(usave));

for t = 1:length(tsave)
    ux(t,2:end) = diff(usave(t,:))/dx;
    uxx(t,3:end) = diff(ux(t,2:end))/dx;
    uxxx(t,4:end) = diff(uxx(t,3:end))/dx;
    uxxxx(t,5:end) = diff(uxxx(t,4:end))/dx;
end    

%% ============== CONSTRUCT I/O DATA ======================================

% Select random grid positions and feed u and its derivatives to network
x_train_index = 5:8:1024;
input = [];
output = [];
for jj = 1:length(x_train_index)
    x_pos = x_train_index(jj)*ones(length(tsave)-1,1);
    k = x_train_index(jj);
    input = [input; usave(1:end-1,k),ux(1:end-1,k),uxx(1:end-1,k),uxxxx(1:end-1,k),x_pos];
    output = [output; usave(2:end,k)];
end    


%% ================== TRAIN NN ============================================
nn = feedforwardnet([15 15 15]);
nn.layers{1}.transferFcn = 'logsig';
nn.layers{2}.transferFcn = 'radbas';
nn.layers{3}.transferFcn = 'tansig';
nn.trainParam.epochs = 2000;
nn.trainParam.max_fail = 1000;
[nn,tr] = train(nn,input.',output.');

% Compute statistics on training dataset
u_net = nn(input.')';

for jj = 1:length(x_train_index)
    pp = jj-1;
    u_net_time(1:70,jj) = u_net(1+70*pp:70*jj);
    output_time(1:70,jj) = output(1+70*pp:70*jj);
end

figure; hold on; grid on
for kk = [10 40 70]
    plot(tsave(2:end),output_time(:,kk),'color',grey,'LineWidth',1.5,'DisplayName',"KS x="+num2str(x_train_index(kk)))
    plot(tsave(2:end),u_net_time(:,kk),'--','LineWidth',1.5,'DisplayName',"NN x="+num2str(x_train_index(kk)))
end
xlabel('time [s]')
ylabel("output u")
legend show

for jj = 1:length(x_train_index)
    RMSE(jj) = rms(output_time(:,jj)-u_net_time(:,jj))/rms(output_time(:,jj))*100;
end
figure; 
plot(x_train_index,RMSE,'*','LineWidth',1.5)
grid on
xlabel("space x")
ylabel('training set RMSE [$\%$]')


%% ========================= VALIDATION ===================================
clear input_val output_real output_net_val_time output_real_time RMSE_val
x_val_index = randperm(1024,length(x_train_index));
x_val_index = sort(x_val_index);
x_val_index = x_val_index(x_val_index ~= x_train_index);
x_val_index = x_val_index(x_val_index>=5 & x_val_index<1024);

for jj = 1:length(x_val_index)
    x_pos = x_val_index(jj)*ones(length(tsave)-1,1);
    k = x_val_index(jj);
    input_val = [usave(1:end-1,k),ux(1:end-1,k),uxx(1:end-1,k),uxxxx(1:end-1,k),x_pos]; 
    output_real = [usave(2:end,k)];

    output_net_val_time(:,jj) = nn(input_val.');
    output_real_time(:,jj) = output_real;
end  


figure; hold on; grid on
for kk = [30 50 90]
    plot(tsave(2:end),output_real_time(:,kk),'color',grey,'LineWidth',1.5,'DisplayName',"KS x="+num2str(x_val_index(kk)))
    plot(tsave(2:end),output_net_val_time(:,kk),'--','LineWidth',1.5,'DisplayName',"NN x="+num2str(x_val_index(kk)))
end
xlabel('time [s]')
ylabel("output u")
legend show

for jj = 1:length(x_val_index)
    RMSE_val(jj) = rms(output_real_time(:,jj)-output_net_val_time(:,jj))/rms(output_real_time(:,jj))*100;
end
figure; 
plot(x_val_index,RMSE_val,'*','LineWidth',1.5)
grid on
xlabel("space x")
ylabel('validation set RMSE [$\%$]')


%% ================= RECONSTRUCT FULL GRID ================================
clear input_val output_real output_net_val_time output_real_time 
clear RMSE_total RMSE_val RMSE_train x_val_index

x_train_index = 5:8:1024;
x_space=5:length(xsave);

for jj = 1:length(x_space)
    x_pos = x_space(jj)*ones(length(tsave)-1,1);
    k = x_space(jj);
    input_val = [usave(1:end-1,k),ux(1:end-1,k),uxx(1:end-1,k),uxxxx(1:end-1,k),x_pos]; 
    output_real = [usave(2:end,k)];
    output_net_val_time(:,jj) = nn(input_val.');
    output_real_time(:,jj) = output_real;
end  
output_net_total = [usave(1,5:end); output_net_val_time];

% Construct plots of the grid
t0 = tsave(1);
figure;
sp(1) = subplot(1,2,1);
surf(tsave-t0,x_space,usave(:,x_space)'),shading interp, colormap("hot"), view(2);
xlim([0 9.8]); ylim([0 1024])
xlabel('time [s]'); 
ylabel('spatial point x');

c=colorbar; zlabel('output u KS equation')
c.Label.String = "output u equation"; c.Label.FontSize = 16.5;
c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';

sp(2) = subplot(1,2,2);
surf(tsave-t0,x_space,output_net_total'),shading interp, colormap("hot"), view(2);
xlim([0 9.8]); ylim([0 1024])
xlabel('time [s]'); 
ylabel('spatial point x');

c= colorbar; zlabel('output u NN')
c.Label.String = "output u NN"; c.Label.FontSize = 16.5;
c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';


pp = 1; vv=1;
for jj = 1:length(x_space)
    RMSE_tot(jj) = rms(output_real_time(:,jj)-output_net_val_time(:,jj))/rms(output_real_time(:,jj))*100;
    if find(x_train_index==x_space(jj),1)
            RMSE_train(pp) = RMSE_tot(jj); pp=pp+1;
    else
            RMSE_val(vv) = RMSE_tot(jj);
            x_val_index(vv) = x_space(jj); vv=vv+1;
    end
end


figure; 
sp(1) = subplot(1,2,1);
plot(x_train_index,RMSE_train,'*','color',grey,'LineWidth',1.5,'MarkerSize',5)
xlabel('spatial point x');
ylabel('RMSE training set [$\%$]')
sp(2) = subplot(1,2,2);
plot(x_val_index,RMSE_val,'*','LineWidth',1,'MarkerSize',5)
xlabel('spatial point x');
ylabel('RMSE validation set [$\%$]')
linkaxes(sp,'x')
xlim([0 1100])

