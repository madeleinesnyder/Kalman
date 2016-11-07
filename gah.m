%% initialize matrixes

t = zeros(1,30);
xpred = zeros(1,30); % initialize the state vector estimate
x = zeros(1,30); % initialize the hidden state vector

%% scale the exponential times
scale = ((3.3*10^5)/2)^(1/29);

%% create the initial taus (t) matrix
t(1,1) = 2;
for i = 1:30
    t(1, i) = t(1, 1) * scale ^ (i - 1);
end

% State Matrices
F = zeros(30,30); % Predicted state transition matrix
for i = 1:30
	for j = 1:30
    	if (i == j)
    		x(i,j) = (1-(1/t(1,i)));
    	else
    		x(i,j) = 0;
        end
    end 
end

P = zeros(30,30); % Predicted state covariance matrix 
for i = 1:30
	for j = 1:30
    	if (i == j)
    		x(i,j) = (10^(-6));
    	else
    		x(i,j) = 0;
        end
    end 
end

c = zeros(1,30);
for i = 1:30
    c(1,i) = (0.001^2)
end

Q = zeros(30,30); % State noise matrix
for i = 1:30
	for j = 1:30
    	if (i == j)
    		x(i,j) = (c/t);
    	else
    		x(i,j) = 0;
        end
    end 
end

%% Prediction Equations

xpred = F*x % Predicted state estimate (x is a 30x1 vector, F is a 30x30 matrix)
Ppred = F*P*(F)' + Q % Prediected estimate covariance

% Observation Matrices
H = ones(30,1) % Observation matrix

R = zeros(30,30) % Observation noise matrix (variances)
for i = 1:30
	for j = 1:30
    	if (i == j)
    		x(i,j) = (0.05^2);
    	else
    		x(i,j) = 0;
        end
    end 
end

z = sum(xpred) + w  % In the model this is y

%% Observation equations
y = z-H*xpred % measurement residual OR [e = y - C*xpred; % error (innovation)] z is observation at time t

S = H*P*(H)' + R %% OR [S = C*Vpred*C' + R]
S_inverse = inv(S);
K = P*(H)'*S_inverse % Kalman gain matrix OR K = Vpred*C'*S_inverse
x = x + K*y 
P = (eye(30)-K*H)*P






