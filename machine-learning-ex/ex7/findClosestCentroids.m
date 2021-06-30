function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% matrix that contains the distance of each training example 
% with respect to each centroid.
distances_matrix = zeros(K, size(X, 1));

for i = 1:K
    centroid_point = centroids(i, :);  
    
    % Computing the Euclidean distance between each training example
    % and the centroids, we can rewrite the Pythagorean theorem as 
    % d=((x_2 - x_1)² + (y_2 - y_1)²) to find the distance between 
    % any two points.
    distances_matrix(i, :) = sqrt(sum((X - centroid_point) .^ 2, 2));    
end

% We choose the indices (centroids) with the smallest distance
[~, idx] = min(distances_matrix);


%%%%% === Different method to find the closest centroids === %%%%%
% matrix that contains the distance of each training example 
% with respect to each centroid.
%distances_matrix = zeros(size(X, 1), K);

%for i = 1:K
    % training examples points
    %x_1 = X(:, 1);
    %y_1 = X(:, 2);
    
    % ith centroid points
    %x_2 = centroids(i, 1);
    %y_2 = centroids(i, 2);  
    
    % Computing the Euclidean distance between each training example
    % and the centroids, we can rewrite the Pythagorean theorem as 
    % d=((x_2 - x_1)² + (y_2 - y_1)²) to find the distance between 
    % any two points.
    %distances_matrix(:, i) = sqrt((x_2 - x_1).^ 2 + (y_2 - y_1) .^ 2);        
%end

% We choose the indices (centroids) with the smallest distance
%[~, idx] = min(distances_matrix');
    
% =============================================================

end

