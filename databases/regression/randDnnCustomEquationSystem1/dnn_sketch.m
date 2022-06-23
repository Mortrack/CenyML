close all; clc; clear;

x_1 = linspace(0, 100, 100);
x_2 = linspace(0, 100, 100);
[x1, x2] = meshgrid(x_1,x_2);

% LAYER 1
Ne_1_1 = tanh(0 - 0.4*x1 + 0.4*x2); % Hiperbolic tangent
Ne_2_1 = -2.34 + 0.09*x1 + 0.77*x2; % 1st order degree
Ne_3_1 = 1./(1+exp(-1.26 - 0.3*x1 + 0.5*x2)); % logistic
Ne_4_1 = (-0.91 +0.04*x1 - 0.0037*x2).^4; % 4th order degree
Ne_5_1 = exp(1.64 - 0.022*x1 - 0.022*x2); % 1st order exponential

% LAYER 2
Ne_1_2 = 0 + Ne_1_1 + 0.1*Ne_2_1 + Ne_3_1 + Ne_4_1 + 2.19*Ne_5_1; % 1st order degree
Ne_2_2 = tanh(0 + Ne_1_1 - 0.1*Ne_2_1 + Ne_3_1 + 0.43*Ne_4_1 + 1.38*Ne_5_1); % Hiperbolic tangent
Ne_3_2 = exp((-0.2 - 1.1*Ne_1_1 + 0.01*Ne_2_1 - 0.02*Ne_3_1 + 0.0074*Ne_4_1 + 0.017*Ne_5_1).^2); % 2nd order exponential
Ne_4_2 = (-0.002 + 0.02*Ne_1_1 - 0.002*Ne_2_1 - 0.3*Ne_3_1 - 0.019*Ne_4_1 + 0.017*Ne_5_1).^6; % 6th order degree

% LAYER 3
Ne_1_3 = 1./(1+exp(1 + 0.4*Ne_1_2 - 0.9*Ne_2_2 - Ne_3_2 - 1.27*Ne_4_2)); % logistic

figure(1)
surf(x1,x2,Ne_1_3);
xlabel('x1')
ylabel('x2')
zlabel('Output')