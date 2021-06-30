function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

% X = m x (n + 1), 
% X = (no_training_examples x (no_features + intercept_feature))
% y = m x 1 = (no_training_examples x 1) dimensional column vector

%%%%% 1.2 Regularized linear regression cost function %%%%%
h_x = X * theta;
sq_error = (h_x - y) .^ 2;
reg_term = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

% computing the vectorized cost function for 
% logistic regression with the regularization term
J = (1 / (2 * m)) * sum(sq_error) + reg_term;

%%%%% 1.3 Regularized linear regression gradient %%%%%
error = (h_x - y);
grad(1) = (1 / m) * (X(:, 1)' * error);

% computing the regularization term to prevent overfitting
extra_term = (lambda / m) * theta(2:end);
% vectorized implementation of the 
% regularized linear regression gradient
grad(2:end) = (1 / m) * (X(:, 2:end)' * error) + extra_term;

% =========================================================================

grad = grad(:);

end
