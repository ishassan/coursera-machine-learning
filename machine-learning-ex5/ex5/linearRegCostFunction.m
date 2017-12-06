function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hypothesis = X * theta;               % size(hypothesis) = [m 1]
error = hypothesis - y;               % size(error) = [m 1]
sumErrorSquare = error' * error;
thetaReg = [0; theta(2:end)];
sumThetaSquare = thetaReg' * thetaReg;

J = (1 / (2.0 * m)) * sumErrorSquare + (lambda / (2.0 * m)) * sumThetaSquare;
sumErrorDerv = error' * X;            % size(sumErrorDerv) = [1 theta.sz]
grad = (sumErrorDerv / m)';           % size(grad) = [theta.sz 1]
grad += (lambda / m) * thetaReg;

%% ==== Iterative solution ====
% for j = 1:length(theta)
%   sumErrorDerv = error' * X(:,j);
%   grad(j,1) = sumErrorDerv / m;
%   grad(j,1) += (lambda / m) * thetaReg(j,1);
% endfor
%% =============================















% =========================================================================

grad = grad(:);

end
