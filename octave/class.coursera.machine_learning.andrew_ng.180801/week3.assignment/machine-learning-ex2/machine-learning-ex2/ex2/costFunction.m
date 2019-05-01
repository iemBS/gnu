function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = X * theta; % transpose of theta * X = theta0 * x0 ... thetaN * xN, nx3 * 3x1 = nx1, each row of vector "z" corresponds to a single row of X.  
g = 1 ./ (1 + (e.^-z)); % put "z" into the sigmoid function, elementwise ops required for this equ to handle scalar or matrix values. 

s = 0;
% summation within the cost function for the logistic regression 
for i=1:m
  s = s + (y(i) * log(g(i))) + ((1 - y(i)) * log(1 - g(i)));
end

J = -(1/m) * s; % cost function for the logistic regrssion, with initial theta this will be 0.693

% now work on the gradient portion of this function. 

grad4theta0Sum = 0;
grad4theta1Sum = 0;
grad4theta2Sum = 0;
% Summation within the gradient equation for the logistic regression, this is not gradient descent equation.
% Notice this does not loop through iterations. We are just doing the sum for all members of the example set just once. 
for i=1:m
  grad4theta0Sum = grad4theta0Sum + ((g(i) - y(i)) * X(i,1));  % (1x1 - 1x1) * 1x1  = nx1, theta0 just equals 1
  grad4theta1Sum = grad4theta1Sum + ((g(i) - y(i)) * X(i,2));  % (1x1 - 1x1) * 1x1  = nx1
  grad4theta2Sum = grad4theta2Sum + ((g(i) - y(i)) * X(i,3));  % (1x1 - 1x1) * 1x1  = nx1
end

grad = (1/m) * [grad4theta0Sum;grad4theta1Sum;grad4theta2Sum];





% =============================================================

end
