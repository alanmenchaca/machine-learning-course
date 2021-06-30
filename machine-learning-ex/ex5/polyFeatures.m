function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% DIMENSIONS
% X' = (1 x m) 
% pVec = (1 x p)
% X_poly = (m x p)

% Creating a column vector of 
% exponentials (1 to p-th power);
pVec = [1:p]';
% X_poly, matrix of polynomial features
X_poly(:) = (X' .^ pVec)';

% =========================================================================

end
