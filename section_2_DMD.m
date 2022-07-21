clear all
clc
close all

run('optdmd-master\optdmd-master\setup.m');

%% OPTDMD parameters
maxiter = 40; % maximum number of iterations
opts = varpro_opts('maxiter',maxiter,'tol',tol,'eps_stall',eps_stall);
tol = 1.0e-6; % tolerance of fit
eps_stall = 1.0e-12; % tolerance for detecting a stalled optimization

%% =================== RAW LYNX HARES DATASET =============================
x1 = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65];
x2 = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25];
X = [x1; x2];

tend = length(x1)-1;
n = tend+1;
dt = 2; t1 = 1845; t2 = 1903;
t = t1:dt:t2;

% Plot raw dataset
figure; 
hold on; grid on;
plot(t,x1,'-*','DisplayName','Hare')
plot(t,x2,'-o','DisplayName','Lynx')
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')

%% ======================== SIMPLE DMD ====================================
r = 2;
imode = 1;
[w,e,b] = optdmd(X,t,r,imode);

% Reconstruct DMD signals
X_dmd = w*diag(b)*exp(e*t);
relerr_r = norm(X_dmd-X,'fro')/norm(X,'fro');

% Plot reconstructed data
figure; 
subplot(1,2,1)
hold on; grid on;
plot(t,x1,'-*','color',grey,'DisplayName','Hare')
plot(t,X_dmd(1,:),'-*','DisplayName','Reconstructed Hare')
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')

subplot(1,2,2)
hold on; grid on;
plot(t,x2,'-o','color',grey,'DisplayName','Lynx')
plot(t,X_dmd(2,:),'-*','DisplayName','Reconstructed Lynx')
legend('show');
xlabel('time [year]');


%% ===================== TIME-DELAY DMD ===================================
PLOT = 1;
time_delay = [2,3,4,5,6];

for jj = 1:length(time_delay)
    % Construct time-delay matrix
    H = [];
    for k = 1:time_delay(jj)
       H = [H; x1(k:end - time_delay(jj) + k); x2(k:end - time_delay(jj) + k)]; 
    end   
    clear t2 t_H
    dt = 2; t1 = 1845; t2 = 1903-(time_delay(jj)-1)*dt;
    t_H = t1:dt:t2;
    
    log.t_H{jj} = t;
    log.H{jj}   = H;

    % Apply DMD to time-delay matrix
    clear r
    r = 2*time_delay(jj);
    imode_init = 1;

    ubc_init = [zeros(r,1); Inf*ones(r,1)];
    lbc_init = [-Inf*ones(r,1); -Inf*ones(r,1)];
    copts_init = varpro_lsqlinopts('lbc',lbc_init,'ubc',ubc_init);
    
    clear w e b
    [w,e,b] = optdmd(H,t_H,r,imode_init,opts,[],[],copts_init);
    
    % Reconstruct dmd signal
    clear X_dmd
    X_dmd = w*diag(b)*exp(e*(t));
    
    log.w{jj}   = w;
    log.e{jj}   = e;
    log.b{jj}   = b;
    log.X_dmd{jj}   = X_dmd;   

end

% PLOT
for jj = 1:length(time_delay)
    if PLOT
        if jj == 1
        figure; 
        axis_plot_rec(1) = subplot(1,2,1); hold on; grid on; box on; legend('show'); xlabel('time [year]'); ylabel('head numbers [thousands]')
        axis_plot_rec(2) = subplot(1,2,2); hold on; grid on; box on; legend('show'); xlabel('time [year]');
        plot(axis_plot_rec(1),t,x1,'-*','color',gray_color(100,:),'LineWidth',1,'DisplayName','Hare')
        plot(axis_plot_rec(2),t,x2,'-*','color',gray_color(100,:),'LineWidth',1,'DisplayName','Lynx')
        end
       
        plot(axis_plot_rec(1),log.t_H{jj},log.X_dmd{jj}(1,:),'-*','DisplayName',['Reconstructed Hare. - delay = ' num2str(time_delay(jj))])
        plot(axis_plot_rec(2),log.t_H{jj},log.X_dmd{jj}(2,:),'-*','DisplayName',['Reconstructed Lynx - delay = ' num2str(time_delay(jj))])
        
    end
    
    if PLOT
        if jj == 1
        figure; 
        axis_plot_eig = axes; 
        hold on; grid on; box on; legend('show'); title('eigenvalues');
        end
        scatter(axis_plot_eig,real(log.e{jj}),imag(log.e{jj}),20*[1:time_delay(jj)*2],'filled','DisplayName',['delay = ' num2str(time_delay(jj))]);
        
    end    
end


%% ======================= BAGGING ======================================== 
clear H
H = [];
for k = 1:time_delay
   H = [H; x1(k:end - time_delay + k); x2(k:end - time_delay + k)]; 
end   
dt = 2; t1 = 1845; t2 = 1903-(time_delay-1)*dt;
t_H = t1:dt:t2;

PLOT_ALL_BAGS = 1;
bag_size = 22;

% Select random samples from the full H matrix
num_bags = 100;      
for kk = 1:num_bags
    index = randperm(length(X)-time_delay+1,bag_size);
    index = sort(index);
    index_set{kk} = index;
end

r = 2*time_delay;
imode = 1;

lbc = [-Inf*ones(r,1); -Inf*ones(r,1)];
ubc = [zeros(r,1); Inf*ones(r,1)];
copts = varpro_lsqlinopts('lbc',lbc,'ubc',ubc);

% Perform iteratively dmd on different bags
for jj = 1:num_bags
    t_bag = t_H(index_set{jj});
    H_bag = H(:,index_set{jj});
    [w_bag{jj},e_bag{jj},b_bag{jj},~,FLAG{jj}] = optdmd(H_bag,t_bag,r,imode,opts,[],[],copts);
end

% Sort eigenvalues to match corresponding values for different bags
autoval = cell2mat(e_bag);

e_sorted{1} = e_bag{1};
w_sorted{1} = w_bag{1};
b_sorted{1} = b_bag{1};
for kk=1:size(autoval,2)
    [~,ind] = sort(abs(autoval(:,kk)));
    e_sorted{kk} = e_bag{kk}(ind);
    w_sorted{kk} = w_bag{kk}(ind,:);
    b_sorted{kk} = b_bag{kk}(ind);
end

autoval2 = cell2mat(e_sorted);

figure; 
scatter(real(autoval2),imag(autoval2))
    
% Compute average
w_bag_vect = w_sorted{1};
e_bag_vect = e_sorted{1};
b_bag_vect = b_sorted{1};
for m = 2:length(w_bag) % numero di bag testate
    w_bag_vect = w_bag_vect + w_sorted{m};
    e_bag_vect = e_bag_vect + e_sorted{m};
    b_bag_vect = b_bag_vect + b_sorted{m};
end
w_bag_mean = w_bag_vect/length(w_bag);
e_bag_mean = e_bag_vect/length(w_bag);
b_bag_mean = b_bag_vect/length(w_bag);

% Reconstruct signal from average decomposition
dt = 2; t1 = 1845; t2 = 1903;
t = t1:dt:t2;
X_dmd_H = w_bag_mean*diag(b_bag_mean)*exp(e_bag_mean*t);
X_dmd_H = X_dmd_H(1:2,:);

relerr_X_rec_H = norm(X_dmd_H-X,'fro')/norm(X,'fro');

% Plot comparison between average model and single bags
figure;
ax1 = subplot(1,2,1);
hold on; grid on;
ax2 = subplot(1,2,2);
hold on; grid on;
X_av_bag = zeros(2,length(t));
for jj =1:length(w_bag)
    X_dmd_bag{jj} = w_bag{jj}*diag(b_bag{jj})*exp(e_bag{jj}*t);
    X_dmd_bag{jj} = X_dmd_bag{jj}(1:2,:);
    
    X_av_bag =  X_av_bag + X_dmd_bag{jj};

    plot(ax1,t,X_dmd_bag{jj}(1,:),'Color',[0.8,0.8,0.8],'HandleVisibility','off')
    plot(ax2,t,X_dmd_bag{jj}(2,:),'Color',[0.8,0.8,0.8],'HandleVisibility','off')
end
X_av_bag = X_av_bag./length(w_bag);

subplot(1,2,1)
plot(ax1,t,x1,'-*','color','b','DisplayName','Hare','MarkerSize',4)
plot(ax1,t,X_av_bag(1,:),'-*','Color','r','DisplayName','Avg Hare','MarkerSize',4)
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')
ylim([-50,150]);

subplot(1,2,2)
plot(t,x2,'-*','color','b','DisplayName','Lynx','MarkerSize',4)
plot(ax2,t,X_av_bag(2,:),'-*','Color','r','DisplayName','Avg Lynx','MarkerSize',4)
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')
ylim([-50,100]);



