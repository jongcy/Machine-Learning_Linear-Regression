function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_len = length(theta);

for iter = 1:num_iters

   
	theta_temp = theta;
	for j = 1:theta_len		
		a = 0;

		for i = 1:m
			a = a + (X(i,:) * theta- y(i,:)) * X(i,j);
		end

		theta_temp(j,:) = theta_temp(j,:) - (alpha/m*a);
	end

	theta = theta_temp;

    J_history(iter) = computeCost(X, y, theta);

end

end