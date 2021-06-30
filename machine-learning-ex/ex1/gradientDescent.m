function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % the predictions it's a m x 1 dimensional column-vector
    h_x = X * theta; 
    error = h_x - y;
    % computing "one step" of gradient descent
    theta1 = theta(1) - alpha * ((1 / m) * (X(:, 1)' * error));
    theta2 = theta(2) - alpha * ((1 / m) * (X(:, 2)' * error));
    % Updating the values of theta simultaneously
    theta = [theta1; theta2];
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
