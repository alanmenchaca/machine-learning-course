function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% We use the previously function to compute the cost function
% for logistic regression and the gradients of J(theta) with
% respect the parameters theta without the regularization term.
[J_temp, grad_temp] = costFunction(theta, X, y);
reg_term = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

% computing the cost function for logistic regression
% with the regularization term to prevent overfitting.
J = J_temp + reg_term;

% computing partial derivates (gradients) of the cost function
reg_term = (lambda / m) * theta(2:end);
grad(1) = grad_temp(1);
grad(2:end) = grad_temp(2:end) + reg_term;

% =============================================================

end
