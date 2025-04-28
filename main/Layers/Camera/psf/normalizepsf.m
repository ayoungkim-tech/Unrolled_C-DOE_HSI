function [psfn, dz_dpsf] = normalizepsf(opts, psf, dz_dpsfn)

% Normalize psf to sum up to 1
% Input:
% psf: tensor containing psfs of each channel (H*W*C)
% Output:
% psf_n: normalized psfs of each channel
% dz_dpn: the derivative of error wrt normalized psf
% dz_dp: the derivative of error wrt psf
%
% Forward:
% psfn = psf/sum(sum(psf)) 
%
% Backward:
% dz_dpsf = dz_dpsfn/sum(sum(psf)) 

nc = reshape([opts.normconst],1,1,[]);
psfn = bsxfun(@rdivide, psf, nc);

if nargout > 1 % Backward
    dz_dpsf = bsxfun(@rdivide, dz_dpsfn, nc);
end