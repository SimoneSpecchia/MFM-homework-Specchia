clear all, close all
addpath('..\00_utils')
grey = [0.5 0.5 0.5];

%% =================== DATASET GENERATION =================================
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; 
r_vect = [10 28 35];
input=[]; output=[]; rho_used = [];

% Generate Lorenz trajectories from random initial conditions
for r = r_vect
    Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                      r * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

    for j=1:70  % training trajectories
        x0(:,j)=30*(rand(3,1)-0.5);
        [t,y] = ode45(Lorenz,t,x0(:,j));
        input=[input; y(1:end-1,:)];
        rho_used = [rho_used; r*ones(height(y(1:end-1,:)),1)];
        output=[output; y(2:end,:)];
    end
end

input = [input rho_used];

%% ======================= TRAINING =======================================
nn = feedforwardnet([10 10]);
nn.layers{1}.transferFcn = 'logsig';
nn.layers{2}.transferFcn = 'purelin';
nn.trainParam.epochs = 2000;
[nn,tr] = train(nn,input.',output.');

%% ======================= VALIDATION =====================================

dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; 
r=40; %17 40

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(2)
subplot(1,2,1)
x0_val=20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0_val);
plot3(x0_val(1),x0_val(2),x0_val(3),'yo','Linewidth',2,'DisplayName','x$_1$(0)')
hold on; grid on
plot3(y(:,1),y(:,2),y(:,3),'Linewidth',1.5,'DisplayName','ODE') 
xlabel('x'); ylabel('y'); zlabel('z')

% Simulate NN
x_input = [x0_val; r];
ynn(1,:)=x0_val;
for jj=2:length(t)
    y0=nn(x_input);
    ynn(jj,:)=y0.'; x_input=[y0; r];
end

% Results visualization
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',1.5,'DisplayName','NN')
legend show

figure(3)
sp(1)=subplot(3,2,1); hold on; grid on; box on
plot(t,y(:,1),'color',grey,'Linewidth',1.5,'DisplayName','ODE')
plot(t,ynn(:,1),'Linewidth',1.5,'DisplayName','NN')
xlabel('time [s]'); ylabel('coordinate x')
legend show
sp(2)=subplot(3,2,3); hold on; grid on; box on
plot(t,y(:,2),'color',grey,'Linewidth',1.5)
plot(t,ynn(:,2),'Linewidth',1.5)
xlabel('time [s]'); ylabel('coordinate y')

sp(3)=subplot(3,2,5); hold on; grid on; box on
plot(t,y(:,3),'color',grey,'Linewidth',1.5)
plot(t,ynn(:,3),'Linewidth',1.5)
xlabel('time [s]'); ylabel('coordinate z')
linkaxes(sp,'x')

ind = 1:length(y);
RMSE_x1 = rms(y(ind,1)-ynn(ind,1))/rms(y(ind,1))*100;
RMSE_y1 = rms(y(ind,2)-ynn(ind,2))/rms(y(ind,2))*100;
RMSE_z1 = rms(y(ind,3)-ynn(ind,3))/rms(y(ind,3))*100;
ind = 1:2/dt;
RMSE_x1_red = rms(y(ind,1)-ynn(ind,1))/rms(y(ind,1))*100;
RMSE_y1_red = rms(y(ind,2)-ynn(ind,2))/rms(y(ind,2))*100;
RMSE_z1_red = rms(y(ind,3)-ynn(ind,3))/rms(y(ind,3))*100;

%% ==================== SET PLOT STYLE ====================================
figure(2), view(-75,15)
figure(3)
subplot(3,2,1), set(gca,'Fontsize',15,'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',15,'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',15,'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',15,'Xlim',[0 8])
subplot(3,2,5), set(gca,'Fontsize',15,'Xlim',[0 8])
subplot(3,2,6), set(gca,'Fontsize',15,'Xlim',[0 8])
legend('Lorenz','NN')