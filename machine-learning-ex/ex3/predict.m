function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then,
%       you can use max(A, [], 2) to obtain the max for each row.
%
%%%%% 2.2 Feedforward propagation and prediction %%%%%
% Theta1 = (no. of hidden units) x (no. of input units + bias unit)
% Theta1 dimensions = 25 x (400 + 1)
% Theta2 = (no. of output units) x (no. of hidden units + bias unit) 
% Theta2 dimensions = 10 x (25 + 1)
a1 = [ones(m, 1), X]; 
z2 = a1 * Theta1'; % (5000 x 401) x (401 x 25)
a2 = sigmoid(z2);  % 5000 x 25
a2 = [ones(m, 1), a2]; % 5000 x (25 + 1)

z3 = a2 * Theta2'; % (5000 x 26) x (26 x 10)
h_x = sigmoid(z3); % 5000 x 10
% the max function returns the maximum element
% in each row and its index.
[~, p] = max(h_x, [], 2); 

% =========================================================================


end
