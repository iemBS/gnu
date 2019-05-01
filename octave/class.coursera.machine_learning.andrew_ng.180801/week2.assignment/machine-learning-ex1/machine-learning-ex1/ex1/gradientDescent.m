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
t = 0;
lowestC = 1000;
a = 0;
b = 0;
c = 0;
J = 0;
lowestJcons = 0;

%X(:,2) = X(:,2)/20; %feature scaling to get x1 between -1 and 1

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
Xtheta = X * theta; % 97x2 times 2x1  = 97x1, [x0 x1] times [theta0;theta1] = theta0x0 + theta1x1
XthetaYXSum1 = 0; %initialize for each iteration
XthetaYXSum2 = 0; %initialize for each iteration

% For each iteration, get sigma, which is the sum of all rows in example set plugged into the equation. 
% For each iteration, sum of all the differences between the hypothetical and actual y values. 
for i = 1:m
  XthetaYXSum1 = XthetaYXSum1 + (Xtheta(i) - y(i));%  scalar = scalar + (1x2 - 1x1)
  XthetaYXSum2 = XthetaYXSum2 + ((Xtheta(i) - y(i)) * X(i,2));%  scalar = scalar + ((1x2 - 1x1) times 1x1)
end
%test: 315.25 = XthetaYXSum1 > 315.25 = sum(-y) where theta0 is zero for the first iteration. 

% amount of change on theta
chg1 = (XthetaYXSum1 * (alpha/m));
chg2 = (XthetaYXSum2 * (alpha/m));
%test: chg1 = 0.032500m = XthetaYXSum1 * alpha >  315.25 = XthetaYXSum1

% Use gradient descent equation to move theta closer to the point where J is lowest. 
% each iteration is a nudge of theta closer to that lowest J point. 
theta = theta - [chg1;chg2]; 
%test: 0.032500 = 0 - chg1  > chg1 = 0.032500

fprintf('iteration %f\n', iter);
fprintf('theta values %f\n', theta);

prevJ = J;
J = computeCost(X, y, theta);


if J > 0 && J < lowestJ
  lowestJ = J;
  lowestJtheta = theta;
  lowestJcons = lowestJcons + 1;
  % show lowest J so far, show if J never goes back up or not 
  % if J only goes down, then alpha too small
  % if J goes down and up, then alpha too big 
  % for alpha: 0.0001=small,0.001=small,0.01=small,0.05=big,0.1=error,
  a = theta(1) - (-3.6303); % distance between two theta0
  b = theta(2) - 1.1664;  % distance between two theta1
  c = sqrt(a^2 + b^2);% pythagorean theorem, distance between two points 
  fprintf('theta0 change %f\n', chg1);
  fprintf('theta1 change %f\n', chg2);
  fprintf('J value %f\n', J);
  fprintf('Lowest consecutive J %f\n', lowestJcons);
  fprintf('theta0 sigma %f\n', XthetaYXSum1);
  fprintf('theta1 sigma %f\n', XthetaYXSum2);
  if (J - prevJ) < 0.001 
    fprintf('!!!!!!!!!!!Convergence\n');% declare convergence if J decreases by less than 10^-3 in one iteration. 
  end
  if c < lowestC
    lowestC = c;
    fprintf('******* closest between target & actual points is %f\n', lowestC);
  end
else
  lowestJcons = 0;
endif

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
