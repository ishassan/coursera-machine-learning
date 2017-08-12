function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% printf("theta = \n");
% size(theta) 3 1
% printf("X = \n");
% size(X) 47 3
% printf("y = \n");
% size(y) 47 1

h = X * theta;
J = sum((h-y) .^ 2) / (2*m);


% =========================================================================

end
