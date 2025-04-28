function [psf, de_dA, de_dphi] = createpsf(opts, d, A, phi, de_dpsf)
%
% Derive psf from amplitude and phase mask parameters
% Formula:
% P = abs(F(A*exp(j*phi)*exp(def(d)))).^2, where def(d) is the depth
%                                          dependent defocus, and F(.)
%                                          is the Fourier transform

%% PSF grid (on lens plane)

p_x = opts(1).p_x;
p_y = opts(1).p_y;

% wavenumber
k = 2*pi./[opts.lambda];
k = reshape(k,1,1,[]);

%% Pupil

if (opts(1).spherical) % Spherical lens (plano convex)
    % Lens radius
    r_lens = opts.r_lens;
    % Central thickness
    D = opts.CT_lens;
    % Lens height function
    h_lens = D - (r_lens - sqrt(r_lens^2 - (p_x.^2 + p_y.^2)));
    % Lens phase
    dn_lambda = reshape([opts.ref_idx_lens] - 1, 1,1,[]);
    phi_lens = bsxfun(@times, k.*1j.*dn_lambda, h_lens);
else % Parabolic lens (approximation)
    f_lens = reshape([opts.f],1,1,[]);
    phi_lens = bsxfun(@times, -k/2.*1j.*1./f_lens,  (p_x.^2 + p_y.^2));
end

psi = 1j * bsxfun(@times, k./2, (1/opts(1).z_i + 1./d));
def = bsxfun(@times, psi, (p_x.^2+p_y.^2));
def = bsxfun(@plus, def, phi_lens);

% Pupil function with defocusing
M = bsxfun(@times, A, exp(1j*phi));
Q = bsxfun(@times, M, exp(def));

%% PSF

FQ = fftshift(fftshift(fft2(Q),1),2);
psf = abs(FQ).^2;

%% Backward

if nargout > 1
    
    % de/dA, de/dphi
    % Detailed derivations will be published
    
    % Derivative of pupil wrt amplitude and phase
    dQ_dA = bsxfun(@times, exp(1j*phi), exp(def));
    dQ_dphi = 1j.*Q; %bsxfun(@times, 1j*A.*exp(1j*phi), exp(def));
    
    N = numel(A(:,:,1,1));

    IFdedpsf = N*ifft2(ifftshift(ifftshift(de_dpsf.*FQ,1),2));
    de_dA = 2*real(IFdedpsf.*conj(dQ_dA));
    de_dphi =  2*real(IFdedpsf.*conj(dQ_dphi));
    
    de_dA = squeeze(sum(de_dA,4));
    de_dphi = squeeze(sum(de_dphi,4));
    
    de_dA = cast(de_dA,'like',de_dpsf);
    de_dphi = cast(de_dphi,'like',de_dpsf);

end


end