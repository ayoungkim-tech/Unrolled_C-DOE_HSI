function [Cam] = Camera_Design(varargin)

% https://www.edmundoptics.com/f/UV-Fused-Silica-Plano-Convex-PCX-Lenses/12410/
% Stock number #36-716

% Using MA-P 1200G Material for refractive index
%% Additional input: color filter size

% default is 15x15
colfiltersize = [15 15];

if nargin>0
    colfiltersize=varargin{1};
end

%% Camera structure
Cam = struct;

%% Desired Wavelengths

lambda = linspace(420,720,31)*1e-9; % nm
nch = numel(lambda);

lambda0 = 510e-9; % nominal wavelength
[~,ch0] = min(abs(lambda-lambda0));
Cam.ch0 = ch0;

%% Lens

% Focus plane for lambda0
z_f0 = 2; %Inf; %% for lambda0, in meters

for nn = 1:nargin
    if strcmpi(varargin{nn},'z_f0')
        z_f0 = varargin{nn+1};
    end
end 

% Focal length specification wavelength
lambda_f = 587.6e-9;
% Lens
f = 35e-3;
% Spherical lens simulation
spherical = false;
% Radius of curvature
r_lens = f*(n_silica(lambda_f*1e+6)-1);
% Center thickness
CT_lens = 2e-3;

% Recalculate the effective focal length of each channel
ref_idx_lens = n_silica(lambda_f*1e+6); % Refractive index at given lambda
% For now, achromatic lens
ref_idx_lens_color = ref_idx_lens .*  ones(size(lambda)); %n_silica(lambda*1e+6); 
f = f * (ref_idx_lens - 1) ./ (ref_idx_lens_color - 1);

% Focal length at lambda0
f0 = f(ch0);
% Lens-to-sensor distance
z_i = (1/f0 - 1/z_f0)^-1; % Based on z_f0
% Object plane (focused) for other wavelengths
z_f = (1./f - 1./z_i).^-1;

%% Sensor parameters

% Sensor
X_x = 6e-6;     
if nargin>3 && ~ischar(varargin{4})
    X_x=varargin{4};
end
X_y = X_x;
N_x = 640;      N_y = 480;
W_x = N_x*X_x;  W_y = N_y*X_y;

%% Mask sampling rate

%p_xx = 15e-6; % X_s in DOE_rotating_psf.m
p_xx = 6e-6; %(lambda(2)-lambda(1)) * z_i / (colfiltersize(1)*X_x);
%p_xx = floor(p_xx*1e+6) * 1e-6; % round in micrometers
if nargin>1
    p_xx=varargin{2};
end          
p_xy = p_xx;

% Aperture
T = 5e-3; %10e-3;

%% Underlying (Low resolution) DOE

p_xx_doe = (lambda(2)-lambda(1)) * z_i / X_x; %30e-6;
p_xx_doe = p_xx * floor(p_xx_doe/p_xx);
if nargin>2
    p_xx_doe=varargin{3};
end

p_xy_doe = p_xx_doe;

%% PSF sizes

p_Nx = ceil(1.2*T./p_xx); % A bit zero padding
is_odd = mod(ceil(p_Nx),2);
p_Nx = is_odd .* p_Nx + (1-is_odd) .* (p_Nx+1);
p_Ny = p_Nx;

% Extend on lens plane
p_wx = p_xx.*p_Nx;
p_wy = p_xy.*p_Ny;

% Desired psf size
psfsize = max(lambda.*z_i / p_xx / X_x);
is_odd = mod(ceil(psfsize),2);
psfsize = is_odd * ceil(psfsize) + (1-is_odd) * (ceil(psfsize)+1);

%% Sampling grid on lens plane

p_y = (-(p_Ny-1)/2:(p_Ny-1)/2) * p_xy ;
p_x = (-(p_Nx-1)/2:(p_Nx-1)/2) * p_xx ;
[p_y,p_x] = meshgrid(p_y,p_x);

% Mask for the lens boundaries
M = (p_x.^2+p_y.^2)  <= (T/2)^2; % Circular Lens

Cam.p_y = p_y;
Cam.p_x = p_x;
Cam.M = M;

%% DOE Sampling grid

p_Nx_doe = ceil(p_wx/p_xx_doe);
is_odd = mod(ceil(p_Nx_doe),2);
p_Nx_doe = is_odd .* p_Nx_doe + (1-is_odd) .* (p_Nx_doe+1);
p_Ny_doe = ceil(p_wy/p_xy_doe);
is_odd = mod(ceil(p_Ny_doe),2);
p_Ny_doe = is_odd .* p_Ny_doe + (1-is_odd) .* (p_Ny_doe+1);

p_y_doe = (-(p_Ny_doe-1)/2:(p_Ny_doe-1)/2) * p_xy_doe ;
p_x_doe = (-(p_Nx_doe-1)/2:(p_Nx_doe-1)/2) * p_xx_doe ;
[p_y_doe,p_x_doe] = meshgrid(p_y_doe,p_x_doe);

% Mask for the lens boundaries
M_doe = (p_x_doe.^2+p_y_doe.^2)  <= (T/2)^2; % Circular Lens

Cam.p_y_doe = p_y_doe;
Cam.p_x_doe = p_x_doe;
Cam.M_doe = M_doe;

%% Maximum defocus and scene depth range

def_max = pi*T/(16*p_xx); %? %p_xx_doe
z_d = pi*T^2./(4*lambda*def_max); % 1/z_d = 1/z+1/z_i-1/f
z_min = (1./z_d - 1./z_i + 1./f).^-1;
z_max = (-1./z_d - 1./z_i + 1./f).^-1;

z_max(z_max<=0) = Inf;
z_min = max(z_min);
z_max = min(z_max);
depth_range = [z_min z_max];

%% Calculate the refractive indices for each channel

ref_idx = real(n_MA_P1200G(lambda*1e+9)); %1.437;

%% Some properties of color filter

% Mu limits 
Cam.mu_low = 390; % in nm, for better optimization (small numbers are tricky)
Cam.mu_high = 750;

% Sigma limits
Cam.sigma_low = ((27) / sqrt(8*log(2))); % in nm, for better optimization (small numbers are tricky)
Cam.sigma_high = ((68) / sqrt(8*log(2)));

% Alpha limits
Cam.alpha_low = 0; % single gaussian case
Cam.alpha_high = 0;

Cam.colfiltersize = colfiltersize;

%% Initial Phase Mask

% For now, zeros
phi0 = 0*M_doe;

Cam.phi0init = single(phi0);

%% DOE upsampling matrices

U1x = upsample1D(p_x_doe(:,1),p_x(:,1),'cubic',3);
U1y = upsample1D(p_y_doe(1,:),p_y(1,:),'cubic',3);
Cam.U1x = U1x;
Cam.U1y = U1y;

%% Store all the values to camera structure
for ch = 1:nch
    
    % Sampling grid for the given wavelength
    X_l = lambda(ch)*z_i/p_wx;
    L_x = X_l*(-(p_Nx-1)/2:(p_Nx-1)/2);
    % Desired sensor grid
    S_x = X_x*(-(psfsize-1)/2:(psfsize-1)/2);
    % Resampling matrix
    D1 = decimate1D(L_x,S_x,3);
    Dtemp = bsxfun(@times, D1, 1./sum(D1, 1)); % normalize
    Dtemp(isnan(Dtemp)) = 0;
    D1 = Dtemp;
    
    % Store values
    Cam(ch).ref_idx = ref_idx(ch);
    Cam(ch).ref_idx_lens = ref_idx_lens_color(ch);
    Cam(ch).z_i = z_i;
    Cam(ch).f = f(ch);
    Cam(ch).z_f = z_f(ch);
    Cam(ch).T = T;
    Cam(ch).X_x = X_x;
    Cam(ch).X_y = X_y;
    Cam(ch).N_x = N_x;
    Cam(ch).N_y = N_y;
    Cam(ch).W_x = W_x;
    Cam(ch).W_y = W_y;
    Cam(ch).numchannel = nch;
    Cam(ch).lambda = lambda(ch);
    Cam(ch).def_max = def_max;
    Cam(ch).normconst = sum(abs(M(:)).^2)*numel(M);
    Cam(ch).D1 = D1;
    Cam(ch).baseKernelSize = size(D1,1);
    %Cam(ch).upsample = upx;
    Cam(ch).p_wx = p_wx;
    Cam(ch).p_wy = p_wy;
    Cam(ch).p_xx = p_xx;
    Cam(ch).p_xy = p_xy;
    Cam(ch).p_xx_doe = p_xx_doe;
    Cam(ch).p_xy_doe = p_xy_doe;
    Cam(ch).depth_range = depth_range;
    Cam(ch).spherical = spherical;
    Cam(ch).r_lens = r_lens;
    Cam(ch).CT_lens = CT_lens;
end




