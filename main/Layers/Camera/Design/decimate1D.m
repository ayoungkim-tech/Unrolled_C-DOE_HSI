function D = decimate1D(x, xq, ord)

%% b - spline
%     B = @(x) (x <= 1).*(x >= 0).*(1 - x) + (x >= -1).*(x <= 0).*(x + 1);
%     B = @(x) 1/2*((x + 3/2).^2).*(-3/2 <= x).*(x < -1/2) - (x.^2 - 3/4).*(-1/2 <= x).*(x <= 1/2) + 1/2*((x - 3/2).^2).*(1/2 < x).*(x <= 3/2);

%% b - spline n-th order
% http://ahay.org/RSF/book/sep/forwd/paper_html/node10.html

% n = ord;
% Mp = @(x) x.*(x > 0);
% Bc = @(x, k, n) Mp(x + (n + 1)./2 - k).^n;
% C = @(n, k) (-1).^k.*factorial(n)./(factorial(k).*factorial(n - k));
% B = @(x) sum(bsxfun(@times, Bc(x, 0:n+1, n), C(n+1, 0:n+1)), 2)./factorial(n);
%
% Bsupp = 3; % better to check

%% lanczos 3

a = ord;
B = @(x) sinc(x).*sinc(x./a).*(abs(x) < a);
Bsupp = a;

%% Coeff

xxq = xq(2)-xq(1);
x = x./xxq;
xq = xq./xxq;

dt = bsxfun(@minus, x(:), xq(:)'); % matrix
valid = abs(dt) <= Bsupp;
valid(:,dt(1,:)>0) = 0;
valid(:,dt(end,:)<0) = 0;
h = dt(valid);
v = B(h); % sample B - spline
P = dt.*0;
P(valid) = v;

% P_K = B(tt)
tt = bsxfun(@minus, xq(:), xq(:)'); % matrix
valid = abs(tt) <= Bsupp;
h = tt(valid);
v = B(h);
Pq = tt.*0;
Pq(valid) = v;

%% pinv = (P'*P)\P'

% D = Pq*pinv(P);
D = P';
D = single(D);

return;



