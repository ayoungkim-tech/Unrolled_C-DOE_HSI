function [phi,de_dphi0] = phi0_to_phi_MS(opts,phi0,de_dphi)

% Nominal channel
ch0 = opts(1).ch0;

% phi = 2*pi*(n-1)*h/lambda, h = lambda0*phi0/(2*pi*(n0-1))
n = reshape([opts.ref_idx],1,1,[]);
lambda = reshape([opts.lambda],1,1,[]);
n0 = n(ch0);
lambda0 = lambda(ch0);
scale = (2*pi*(n-1)./lambda) * (lambda0/(2*pi*(n0-1)));

phi = bsxfun(@times, phi0, scale);

if nargout > 1
    de_dphi0 = bsxfun(@times, de_dphi, scale);
    de_dphi0 = sum(sum(de_dphi0,3),4);
end

end