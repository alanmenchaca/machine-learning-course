function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%%%%%%% 1.3.1 Vectorizing the cost function %%%%%%%
% n + 1 = number of features plus the intercept term
% X = m x (n + 1) dimensional matrix
% theta = (n + 1) x 1 dimensional column-vector
z = X * theta;    
% h_x = (m x (n + 1)) x ((n + 1) x 1) = m x 1 
% h_x = m x 1 dimensional column-vector
h_x = sigmoid(z); 

% Vectorizing the cost function for logistic regression
J = (1 / m) * (-y' * log(h_x) - (1 - y)' * (log(1 - h_x)));

% Computing the gradient of the 
% (unregularized) logistic regression cost
grad = (1 / m) * (X' * (h_x - y));

%%%%%%% 1.3.2 Vectorizing the gradient %%%%%%%
theta_temp = theta;
theta_temp(1) = 0; % because we don't add anything for j = 0
regularization_term = (lambda / m) * theta_temp;
grad = grad + regularization_term;

%%%%%%% 1.3.3 Vectorizing regularized logistic regression %%%%%%%  
regularization_term = (lambda / (2 * m)) * sum(theta_temp .^ 2);
J = J + regularization_term;

% =============================================================

grad = grad(:);

end
