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
for i = 1:length(idx)
  x = X(i, :);
  min_cost = 1e6;
  best_cent = 0;
  for cent_idx = 1:K
      centroid = centroids(cent_idx, :);
      cost = (norm(x - centroid))^2;
      if(min_cost > cost)
          min_cost = cost;
          best_cent = cent_idx;
      end
  end
  idx(i) = best_cent;
end





% =============================================================

end
