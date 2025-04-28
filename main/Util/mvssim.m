%% MvSSIM
% https://www.sciencedirect.com/science/article/pii/S0925231217312018

function output = mvssim(X,Y, varargin)
% alpha[1]          positive exponent that adjust the relative importance of the compontents,
%                   if it is not given, alpha is 1.
% beta[2]          positive exponent that adjust the relative importance of the compontents,
%                   if it is not given, beta is 1.
% gamma[3]          positive exponent that adjust the relative importance of the compontents,
%                   if it is not given, gamma is 1.
% C1 [4]            constants for luminance similarity,
%                   if it is not given, C1 is 0.
% C2 [5]            constants for covariance similarity,
%                   if it is not given, C2 is 0.
% C3 [6]            constants for spatial structural similarity,
%                   if it is not given, C3 is 0.
%
% Example:          MvSSIM = mvssim(Iin, Iout);
%                   MvSSIM = mvssim(Iin, Iout, 0.9, 0.8,0.9, 0.01, 0.03, 0.01);
%                   Here, Iin and Iout size are (n1,n2,Q).
%
%% Positive exponents that adjust the relative importance of the compontents
if nargin > 2
    alpha = varargin{1};
else
    alpha = 1;
end

if nargin > 3
    beta = varargin{2};
else
    beta = 1;
end

if nargin > 4
    gamma = varargin{3};
else
    gamma = 1;
end

%% Constants for each similarity calculation
if nargin > 5
    C1 = varargin{4};
else
    C1 = 0;
end

if nargin > 6
    C2 = varargin{5};
else
    C2 = 0;
end

if nargin > 7
    C3 = varargin{6};
else
    C3 = 0;
end

%% Calculate MvSSIM
% number of spectral bands Q:
Q = size(X,3);
% the sample means X_bar, Y_bar:
X_bar = zeros(Q,1);
Y_bar = zeros(Q,1);
for i = 1:Q
    X_bar(i) = mean(X(:,:,i),'all');
    Y_bar(i) = mean(Y(:,:,i),'all');
end

n1 = size(X,1);
n2 = size(X,2);
N = n1*n2;
X_N = zeros(Q,Q,N);
Y_N = zeros(Q,Q,N);
XY_N = zeros(Q,Q,N);
for i = 1:n1
    for j = 1:n2
        ind = n1*(i-1)+j;
        X_q = squeeze(X(i,j,:))-X_bar;
        X_N(:,:,ind) = X_q.*X_q.';
        Y_q = squeeze(Y(i,j,:))-Y_bar;
        Y_N(:,:,ind) =  Y_q.*Y_q.';
        XY_N(:,:,ind) = X_q.*Y_q.';
    end
end

% the sample covariance matrices Sigma_X, Sigma_Y:
Sigma_X = mean(X_N,3);
Sigma_Y = mean(Y_N,3);

% the sample cross-covariance matrix Sigma_XY
Sigma_XY = mean(XY_N,3);

output = (l(X_bar,Y_bar,C1)^alpha)*(c(Sigma_X,Sigma_Y,C2)^beta)*(s(Sigma_X,Sigma_Y,Sigma_XY,C3)^gamma);
end

%% Three similarities (luminance, covariance(contrast), spatial structure)
% luminace similarity
function output = l(X_bar,Y_bar,C1)
output = (2.*dot(X_bar,Y_bar)+C1)./(dot(X_bar,X_bar)+dot(Y_bar,Y_bar)+C1);
end

% covariance similarity
function output = c(Sigma_X,Sigma_Y,C2)
lambda_s = sum(svd(Sigma_X));
d_s = sum(svd(Sigma_Y));
output = (2.*sqrt(lambda_s).*sqrt(d_s)+C2)./(lambda_s+d_s+C2);
end

% spatial structure similarity
function output = s(Sigma_X,Sigma_Y,Sigma_XY,C3)
output = mean((diag(Sigma_XY)+C3)./(sqrt(diag(Sigma_X)).*sqrt(diag(Sigma_Y))+C3));
end