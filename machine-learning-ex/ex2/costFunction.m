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

% m = number of training examples
% n + 1 = number of features plus the intercept term
% X = m x (n + 1) dimensional matrix
% theta = (n + 1) x 1 dimensional column-vector
z = X * theta;    
h_x = sigmoid(z); 

% Computing the cost function (scalar) for logistic regression
J = (1 / m) * (-y' * log(h_x) - (1 - y)' * (log(1 - h_x)));

% computing the gradients of the cost function
% with respect to the parameters theta
grad = (1 / m) * X' * (h_x - y);

% =============================================================

end
