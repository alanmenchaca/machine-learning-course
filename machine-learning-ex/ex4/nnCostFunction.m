function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%% 1.3 Feedforward and cost function %%%%%
% a_1 - (no_training_examples x (input_layer_size + 1))
% a_1 - (m x (n + 1)) = 5000 x 401
a_1 = [ones(m, 1), X];
% z_2 - (no_training_examples x (hidden_layer_size + 1))
% z_2 - (m x (hidden_layer_size + 1)) = 5000 x 26
z_2 = a_1 * Theta1'; 
a_2 = sigmoid(z_2);

% a_2 - (no_training_examples x (hidden_layer_size + 1))
% a_2 - (m x (hidden_layer_size + 1)) = 5000 x 26
a_2 = [ones(m, 1), a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
h_x = a_3;

y_vec = ([1: num_labels] .* ones(m, num_labels)) == y;

J = (1 / m) * sum(sum((-y_vec .* log(h_x) - (1 - y_vec) .* log(1 - h_x))));

%%%%% 1.4 Regularized cost function %%%%%
Theta1_reg_term = sum(sum(Theta1(:, 2:end) .^ 2));
Theta2_reg_term = sum(sum(Theta2(:, 2:end) .^ 2));

regularized_term = (lambda / (2 * m)) * (Theta1_reg_term + Theta2_reg_term);
J = J + regularized_term;

%%%%% 2.3 Backpropagation %%%%%
% delta_3 = (no_training_examples x num_labels) = (5000 x 10)
delta_3 = a_3 - y_vec;
% Theta2 = (num_labels x (hidden_layer_size + 1)) = (10 x 26)
% Adding bias node for z_2
z_2 = [ones(m, 1), z_2];

delta_2 = (delta_3 * Theta2) .* sigmoidGradient(z_2);
% delta_2 = (no_training_examples x hidden_layer_size) = 5000 x 25
% Removing bias node for delta_2
delta_2 = delta_2(:, 2:end);

% Theta1_grad = (hidden_layer_size x (input_layer_size + 1)) = (25 x 401)
% Theta2_grad = (num_labels x (hidden_layer_size + 1)) = (10 x 26)
Theta1_grad = (1 / m) * (Theta1_grad + (delta_2' * a_1));
Theta2_grad = (1 / m) * (Theta2_grad + (delta_3' * a_2));
% (delta_2' * a_1) = ((25 x 5000) * (5000 x 401)) = 25 x 401
% (delta_3' * a_2) = ((10 x 5000) * (5000 x 26)) = 10 x 26

%%%%% 2.5 Regularized neural networks %%%%%
% We do not regularize the first column of Theta1_grad 
% which is used for the bias term.
Theta1_grad_regularized = (lambda / m) .* Theta1(:, 2:end);
Theta2_grad_regularized = (lambda / m) .* Theta2(:, 2:end);

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1_grad_regularized;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2_grad_regularized;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
