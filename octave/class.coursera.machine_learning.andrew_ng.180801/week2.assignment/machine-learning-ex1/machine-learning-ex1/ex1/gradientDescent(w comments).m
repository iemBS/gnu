function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%iter_sigma1 = 0;
%iter_sigma2 = 0;
%sigma1 = 0;
%sigma2 = 0;

lowestJ = 1000;
lowestJtheta = zeros(2,1);
Xtheta = zeros(97,1);
XthetaY = zeros(97,1);
XthetaYX = zeros(97,2);
XthetaYXSum1 = 0;
XthetaYXSum2 = 0;
t = 0;
lowestC = 1000;
a = 0;
b = 0;
c = 0;


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
%iter_sigma1 = ((X(iter,1) * theta(1,1))- y(iter,1)) * X(iter,1); 
%iter_sigma2 = ((X(iter,2) * theta(2,1))- y(iter,1)) * X(iter,2);

%iter_sigma1 = ((X(1:m,1) * theta(1,1))- y(1:m,1)); 
%iter_sigma2 = ((X(1:m,2) * theta(2,1))- y(1:m,1)) * X(1:m,2)';

Xtheta = X * theta; % 97x2 times 2x1  = 97x1, [x0 x1] times [theta0;theta1] = theta0x0 + theta1x1

% Get sigma, which is the sum of all rows in example set plugged into the equation. 
for i = 1:m
  XthetaYXSum1 = XthetaYXSum1 + (Xtheta(i,:) - y(i,:));%  scalar = scalar + (1x2 - 1x1)
  XthetaYXSum2 = XthetaYXSum2 + ((Xtheta(i,:) - y(i,:)) * X(i,2));%  scalar = scalar + ((1x2 - 1x1) times 1x1)
  %XthetaY = Xtheta(i,:) - y(i,:); % 97x1(1x1) - 97x1(1x1) = 97x1
  %t = X(i,:)'; % 1x2 change to 2x1, [x0,x1] > x0 in 1st row & x1 in 2nd row
  %XthetaYX(i,:) = XthetaY(i,:) * t; % 97x1(1x1) times 2x1 = 97x2(2x1)
  %XthetaYXSum = XthetaYXSum + XthetaYX % 97x1 times 1x2 = 97x2
%sigma1 = sigma1 + iter_sigma1(i,1);
%sigma2 = sigma2 + iter_sigma2(i,1);
end

% amount of change on theta
chg1 = (XthetaYXSum1 * (alpha/m));
chg2 = (XthetaYXSum2 * (alpha/m));

% Use gradient descent equation to move theta closer to the point where J is lowest. 
% each iteration is a nudge of theta closer to that lowest J point. 
theta = theta - [chg1;chg2]; 
%theta = round(10000 * theta) / 10000; %round to four decimal places



%theta(1,1) = theta(1,1) - ((sigma1 * alpha)/m);
%theta(2,1) = theta(2,1) - ((sigma2 * alpha)/m); 

J = computeCost(X, y, theta);

if J < lowestJ
  lowestJ = J;
  lowestJtheta = theta;
  % show lowest J so far, show if J never goes back up or not 
  % if J only goes down, then alpha too small
  % if J goes down and up, then alpha too big 
  % for alpha: 0.0001=small,0.001=small,0.01=small,0.05=big,0.1=error,
  a = theta(1) - (-3.6303); % distance between two theta0
  b = theta(2) - 1.1664;  % distance between two theta1
  c = sqrt(a^2 + b^2);% pythagorean theorem, distance between two points 
  fprintf('iteration %f\n', iter);
  fprintf('theta0 change %f\n', chg1);
  fprintf('theta1 change %f\n', chg2);
  fprintf('theta values %f\n', theta);
  fprintf('J value %f\n', J);
  if c < lowestC
    lowestC = c;
    fprintf('******* closest point %f\n', lowestC);
    if lowestC < 0.01 %usually 0.15
      fprintf('!!!!!!!!!!!!!!!!!!Reached Convergence zone\n');
      return;
    end
  end
  if abs(chg1) < 0.001 || abs(chg2) < 0.001
    fprintf('%f is less than 10^-3 change for theta\n', chg1);
    fprintf('%f is less than 10^-3 change for theta\n', chg2);
  end
endif



theta = [-3.6303;1.1664];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
