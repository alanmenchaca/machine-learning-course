

function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% m = number of training examples
% n + 1 = number of features plus the intercept term (theta_0)
% X = m x (n + 1) dimensional matrix
% theta = (n + 1) x 1 dimensional column-vector
% h_x = (m x (n + 1)) x ((n + 1) x 1) = m x 1

% h_x is given by the linear model 
h_x = X * theta; % m x 1 dimensional column-vector
% The square error is given by the square difference
% between the predicted values h_x and the real values y.
sq_error = (h_x - y) .^ 2; 

% computing the cost function:
J = (1 / (2 * m)) * sum(sq_error);

% =========================================================================

end
