clear 
close all
clc

%% ================== LOAD RS GRID ========================================
load("reaction_diffusion_big.mat");

figure
imagesc(u(:,:,1));

figure
imagesc(v(:,:,1));

%% ===================== CONSTRUCT NN I/O =================================
clear x_temp x_col
X = [];
x_select = 100:200;
y_select = 100:200;
t_train = 1:150;

for k = 1:length(t)
    x_col = [];
    for xx = x_select
        for yy = y_select
            x_temp = [u(xx,yy,k);v(xx,yy,k)];
            x_col = [x_col;x_temp];            
        end
    end
    X = [X,x_col];
end

[coeff,Z_pca,~,~,variance,mu] = pca(X');

idx = find(cumsum(variance)>95,1);

Z_red = Z_pca(:,1:idx);

input = Z_red(t_train,:);
output = Z_red(t_train + 1,:);

%% ================== TRAIN NN ============================================
nn = feedforwardnet([10 10]);
nn.layers{1}.transferFcn = 'logsig';
nn.layers{2}.transferFcn = 'purelin';
nn.trainParam.epochs = 1500;
nn.trainParam.max_fail = 1000;
[nn,tr] = train(nn,input.',output.');

%% ====================== VALIDATION ======================================
t_validation = 150;


% Transform in z and advance the solution with the NN
z_val_in = (X(:,t_validation)'- mu)*coeff(:,1:idx);
z_val_out = nn(z_val_in');

% Transform back in original coordinates
x_val_out = z_val_out'*coeff(:,1:idx)' + mu;
x_real = X(:,t_validation)';

u_val = zeros(size(u(:,:,1)));
v_val = zeros(size(v(:,:,1)));
k = 1;
for xx = x_select
    for yy = y_select

        u_val(xx,yy) = x_val_out(k);
        k = k+1;
        v_val(xx,yy) = x_val_out(k);
        k = k+1;           
    end
end

figure;
subplot(2,2,1)
surf(x_select,y_select,u_val(x_select,y_select)),shading interp, colormap("hot"), view(2);
subplot(2,2,2)
surf(x_select,y_select,u(x_select,y_select,t_validation + 1)),shading interp, colormap("hot"), view(2);
subplot(2,2,3)
surf(x_select,y_select,v_val(x_select,y_select)),shading interp, colormap("hot"), view(2);
subplot(2,2,4)
surf(x_select,y_select,v(x_select,y_select,t_validation + 1)),shading interp, colormap("hot"), view(2);

%% ======================= COMPUTE ERROR ==================================

u_error = [];
v_error = [];
t_select = 160:length(t)-1;

% Advance the solution for each validation time
for t_validation = t_select

    z_val_in = (X(:,t_validation)'- mu)*coeff(:,1:idx);
    
    z_val_out = nn(z_val_in');
    
    x_val_out = z_val_out'*coeff(:,1:idx)' + mu;
    x_real = X(:,t_validation+1)';
    
    % Compute error for each time instant as the average over the entire
    % grid of the errors for each cell
    u_avg = mean(abs((x_real(1:2:end-1)-x_val_out(1:2:end-1))./abs(x_real(1:2:end-1))),'all');
    v_avg = mean(abs((x_real(2:2:end)-x_val_out(2:2:end))./abs(x_real(2:2:end))),'all');
    u_error = [u_error,u_avg];
    v_error = [v_error,v_avg];
end    

figure
subplot(1,2,1)
scatter(t_select,u_error.*100,20,'filled');
xlabel('time [s]')
ylabel('Average grid error [%]')
subplot(1,2,2)
scatter(t_select,v_error.*100,20,'filled');
xlabel('time [s]')



