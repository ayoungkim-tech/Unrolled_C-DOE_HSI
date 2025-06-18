function [mu, sigma, colfilt] = net2colfilt(net)
% Compute colorfilter parameters from the trained network
% Output: Pysical values (nm) of mu and sigma along with color filter
% Example: [mu, sigma,colfilt] = net2colfilt(net);

% Camera parameters
opts = net.layers(1).block.opts;

% Wavelength
lambda = cat(3,opts.lambda).*1e+9;

% Mu range
mu_low = opts(1).mu_low;
mu_high = opts(1).mu_high;
mu_range = mu_high-mu_low;

% Sigma range
sigma_low = opts(1).sigma_low;
sigma_high = opts(1).sigma_high;
sigma_range = sigma_high-sigma_low;

%
norm = 'sum';
activation = 'tanh';

%% Load colorfilter
mu = net.params(2).value(:,:,1);
sigma = net.params(3).value(:,:,1);

switch lower(activation)
    case 'sin'
        mu = 1/2 * (sin(mu)+1);
        sigma = 1/2 * (sin(sigma)+1);
    case 'tanh'
        mu = 1/2 * (tanh(mu)+1);
        sigma = 1/2 * (tanh(sigma)+1);
    otherwise % error
        error('Unknown activation %s.', obj.activation);
end

% Threshold
mu = min(max(mu, 0), 1);
sigma = min(max(sigma, 0), 1);

% Change the range to physical values
mu = mu_range.*mu + mu_low;
sigma = sigma_range.*sigma + sigma_low;

% variance
sigma_2 = sigma.^2;

% Color response
color_response = exp(-bsxfun(@minus,mu,lambda).^2./(2.*sigma_2));

% Normalize
color_response_n = normalize_ch_v2(color_response,norm);
colfilt = color_response_n;