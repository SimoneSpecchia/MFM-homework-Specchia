clc
clear all

%% ======================= DATASET ========================================
x1 = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65];
x2 = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25];
X = [x1; x2];

time = 1845:2:1903;

grey = [0.4 0.4 0.4];


%% ========================= DERIVATIVE ===================================

[m,n] = size(X);
FF = n-1; 
dt = 1;
Xdot= zeros(m,FF);
for kk = 1:m %explore rows = animal
    for jj = 2:FF %explore timeseries
        Xdot(kk,jj-1) = (X(kk,jj+1)-X(kk,jj-1))/2/dt;
    end
end

%% ====================== LEAST SQUARES ===================================
X_red = X(:,2:FF+1);

X1 = X_red(1,1:end)';
X2 = X_red(2,1:end)';
X1X2 = X1.*X2;
O = zeros(FF,1);

A = [X1 -X1X2 O O; O O X1X2 -X2];
Xdot_tmp = [Xdot(1,:)'; Xdot(2,:)'];

% Different methods to solve LS
% phi = lsqr(A,Xdot_tmp); % MATLAB solution
phi = A\Xdot_tmp; % closed-loop formula linear regression in the unknown parameters


%% ==================== SIMULATE OUTPUT ===================================
Xdot_hat = A*phi;
Xdot1_hat = Xdot_hat(1:FF)';
Xdot2_hat = Xdot_hat(FF+1:end)';

pp = length(Xdot2_hat);
figure; hold on;
plot(time(1:pp),Xdot(1,:),'-*','DisplayName','Hare')
plot(time(1:pp),Xdot1_hat(1,:),'-*','DisplayName','L-V Hare')
plot(time(1:pp),Xdot(2,:),'-o','DisplayName','Lynx')
plot(time(1:pp),Xdot2_hat(1,:),'-o','DisplayName','L-V Lynx')
legend show
xlabel('time [years]')
ylabel('Xdot')

clear x1_hat x2_hat
x1_hat(1) = x1(1);
x2_hat(1) = x2(1);
Ts = dt;
% 
for kk = 1:length(x1)-1
    x1_hat(kk+1) = x1_hat(kk) + Ts*(phi(1)*x1_hat(kk) - phi(2)*x1_hat(kk)*x2_hat(kk));
    x2_hat(kk+1) = x2_hat(kk) + Ts*(phi(3)*x1_hat(kk)*x2_hat(kk) -phi(4)*x2_hat(kk));
end

% Plot results
figure; 
subplot(1,2,1)
hold on; grid on;   
plot(time,x1,'-*','color',grey,'DisplayName','Hare','MarkerSize',4)   
plot(time,x1_hat,'-*','DisplayName','LV Hare','MarkerSize',4)   
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')

subplot(1,2,2)
hold on; grid on;  
plot(time,x2,'-*','color',grey,'DisplayName','Lynx','MarkerSize',4)   
plot(time,x2_hat,'-*','DisplayName','LV Lynx','MarkerSize',4)
legend('show');
xlabel('time [year]');
