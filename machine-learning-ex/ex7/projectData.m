function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% U = (n x n) ---> U = (2 x 2) 
% U has 2 dimension or 2 components
% We reduce U from 2 components to 1 component,
% choosing the first column (n x 1)

% U_reduce = (n x 1) 1 dimension, 1 component
% U_reduce = (n x 1) ---> U_reduce = (2 x 1)
U_reduce = U(:, 1:K); 

% We compute the projection of the normalized inputs X into
% the reduced dimensional space spanned by the first K columns
% of U. It returns the projected examples in Z.

% Z = X * U_reduce = (50 x 2) * (2 x 1) = (50 x 1)
% We reduce the dimensions of Z, which now is Z = (50 x 1)
Z = X * U_reduce;

% =============================================================

end