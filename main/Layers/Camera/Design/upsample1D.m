function U = upsample1D(x, xq, method, ord)

% x:input
% xq:output
% method:lanczos or bicubic
% ord: lanczos order (2 or 3)

%% lanczos or cubic

switch method
    case 'lanczos'
        a = ord;
        B = @(x) lanczos3(x); %sinc(x).*sinc(x./a).*(abs(x) < a);
        Bsupp = a;
    case {'cubic', 'bicubic'}
        B = @(x) cubic(x);  
        Bsupp = 3;
end    

%% Coeff

xx = x(2)-x(1);
x = x./xx;
xq = xq./xx;

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

%  U = Pq*pinv(P);
U = P';
U = single(U);

return;

end


function f = cubic(x)
% See Keys, "Cubic Convolution Interpolation for Digital Image
% Processing," IEEE Transactions on Acoustics, Speech, and Signal
% Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.

absx = abs(x);
absx2 = absx.^2;
absx3 = absx.^3;

f = (1.5*absx3 - 2.5*absx2 + 1) .* (absx <= 1) + ...
    (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) .* ...
    ((1 < absx) & (absx <= 2));
end

function f = lanczos3(x)
% See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990,
% pp. 157-158.

f = (sin(pi*x) .* sin(pi*x/3) + eps) ./ ((pi^2 * x.^2 / 3) + eps);
f = f .* (abs(x) < 3);
end

