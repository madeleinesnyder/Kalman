%% Temporal Distinctiveness Theory (Capitals are the second X run)

t = zeros(1,30);
xpred = ones(1,30); % initialize the state vector estimate
Xpred = ones(1,30); % initialize the state vector estimate

x = randn(1,30)*0.05 + 1;% gaussian mu 1
X = randn(1,30)*0.05 + 1;% gaussian mu 1

sumx = sum(x);
sumX = sum(X);

for i = 1:30
    x(i) = x(i)/sumx;
    X(i) = X(i)/sumX;
end

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
    F(i,i) = (1-(1/t(1,i)));
end

% Predicted state covariance matrix 
Ppred = eye(30,30)*(10^(-6));
ppred = eye(30,30)*(10^(-6));

Q = zeros(30,30); % State noise matrix
for i = 1:30
    Q(i, i) = 0.001^2 / t(1,i);
end


% Observation Matrices
H = ones(30,1); % Observation matrix

R = zeros(30,30); % Observation noise matrix (variances)
for i = 1:30
    R(i,i) = (0.05^2);
end
j = 1
while j <= 10000

    allx(:,j) = x;
    allX(:,j) = X;
    allxpred(:,j) = xpred;
    allXpred(:,j) = Xpred;
    xpred = (F*xpred')'; % Predicted state estimate (x is a 30x1 vector, F is a 30x30 matrix)
    Xpred = (F*Xpred')';
    ppred = F*ppred*(F)' + Q;
    Ppred = F*Ppred*(F)' + Q; % Prediected estimate covariance
    z = sum(x) + R(1,1); % In the model this is y
    Z = z;
    if (j > 1000 && mod(j, 100) ~= 0)
        Z = randn(1,1)*17 + 10 + R(1,1); % In the model this is y
    end
    y = z - H'*xpred'; % measurement residual OR [e = y - C*xpred; % error (innovation)] z is observation at time t
    Y = Z - H'*Xpred';
    s = H'*ppred*H + R; %% OR [S = C*Vpred*C' + R]
    S = H'*Ppred*H + R;
    k = H'*ppred'*inv(s); % Kalman gain matrix OR K = Vpred*C'*S_inverse
    K = H'*Ppred'*inv(S);
    xpred = xpred + k*y; 
    Xpred = Xpred + K*Y;
    
    for i = 1:30
        err(1,i) = randn(1,1)*(sqrt((0.001^2)/t(i)));
    end
    x = (F*x')' + err;
    
    minXpred = min(Xpred);
    for i = 1:30
        Xpred(i) = Xpred(i) + minXpred;
    end
    
    sumXpred = sum(Xpred);
    sumxpred = sum(xpred);
    sumx = sum(x);
    
    for i = 1:30
        xpred(i) = xpred(i)/sumxpred;
        Xpred(i) = Xpred(i)/sumXpred;
        x(i) = x(i)/sumx;
    end
    Ppred = (eye(30)-K*H)*Ppred;
    ppred = (eye(30)-k*H)*ppred;

    j = j + 1;
end