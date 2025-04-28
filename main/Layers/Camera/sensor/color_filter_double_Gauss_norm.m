function [color_response,de_dmu1, de_dmu2, de_dsigma1, de_dsigma2, de_dalpha] = color_filter_double_Gauss_norm(opts,mu1,sigma1, mu2, sigma2,alpha, de_dcol)

% double Gauss, normalized
% mu (nxn): mean
% sigma (nxn): standard deviation
% de_d: the derivative of error with respect to the correspondent parameter
% de_dcol: the derivative of the actual color filter
% 
% Careful: It does not normalize the color filter itself, do it outside

% wavelength
lambda = cat(3,opts.lambda).*1e+9;

% Threshold

% mu
mu_low = opts(1).mu_low;
mu_high = opts(1).mu_high;
mu_range = mu_high-mu_low;

% We need this part for backpropagation
mu1_der = (mu1 > 0 & mu1 < 1);
mu2_der = (mu2 > 0 & mu2 < 1);

% Thresholding for mu
mu1 = min(max(mu1, 0), 1);
mu1 = mu_range.*mu1 + mu_low; %'unnormalize'
mu2 = min(max(mu2, 0), 1);
mu2 = mu_range.*mu2 + mu_low; %'unnormalize'

% Calculate based on FWHM from 31nm to 300nm
sigma_low = opts(1).sigma_low;
sigma_high = opts(1).sigma_high;
sigma_range = sigma_high-sigma_low;

% We need this part for backpropagation
sigma1_der = (sigma1 > 0 & sigma1 < 1);
sigma2_der = (sigma2 > 0 & sigma2 < 1);

% Thresholding for sigma
sigma1 = min(max(sigma1, 0), 1);
sigma1 = sigma_range.*sigma1 + sigma_low; %'unnormalize'
sigma2 = min(max(sigma2, 0), 1);
sigma2 = sigma_range.*sigma2 + sigma_low; %'unnormalize'

% alpha
alpha_low = opts(1).alpha_low; % single gaussian case
alpha_high = opts(1).alpha_high;
alpha_range = alpha_high-alpha_low;

% We need this part for backpropagation
alpha_der = (alpha > 0 & alpha < 1);

% Thresholding for alpha
alpha = min(max(alpha, 0), 1);
alpha = alpha_range.*alpha + alpha_low; %'unnormalize'

% variance
sigma1_2 = sigma1.^2;
sigma2_2 = sigma2.^2;

% responses
color_response1 = exp(-bsxfun(@minus,mu1,lambda).^2./(2.*sigma1_2));
color_response2 = exp(-bsxfun(@minus,mu2,lambda).^2./(2.*sigma2_2));

% final
color_response = color_response1 + alpha.*color_response2;

if nargout>1
   
   % derivative of two different filters
   de_dcol1 = de_dcol;
   de_dcol2 = alpha.*de_dcol;
   
   % First, mu 
   de_dmu1 = de_dcol1 .* color_response1 .* bsxfun(@minus,lambda,mu1) ./ sigma1_2; % Derivative of gauss
   de_dmu1 = mu_range.*de_dmu1; % derivative of 'unnormalization'
   de_dmu1 = de_dmu1 .* mu1_der; % Derivative of thresholding
   
   % Second, mu
   de_dmu2 = de_dcol2 .* color_response2 .* bsxfun(@minus,lambda,mu2) ./ sigma2_2; % Derivative of Gauss
   de_dmu2 = mu_range.*de_dmu2; % derivative of 'unnormalization'
   de_dmu2 = de_dmu2 .* mu2_der; % Derivative of thresholding
   
   % Sum
   de_dmu1 = mean(de_dmu1,3);
   de_dmu2 = mean(de_dmu2,3);
   
   % First, sigma
   de_dsigma1_2 = de_dcol1 .* color_response1 .* (bsxfun(@minus,mu1,lambda).^2 ./ (2.*sigma1_2.^2)); % Derivative of Gauss
   de_dsigma1 = de_dsigma1_2 .* 2 .* sigma1; % Derivative of square
   de_dsigma1 = sigma_range.*de_dsigma1; % derivative of 'unnormalization'
   de_dsigma1 = de_dsigma1 .* sigma1_der; % Derivative of thresholding
   
   % Second, sigma
   de_dsigma2_2 = de_dcol2 .* color_response2 .* (bsxfun(@minus,mu2,lambda).^2 ./ (2.*sigma2_2.^2)); % Derivative of Gauss
   de_dsigma2 = de_dsigma2_2 .* 2 .* sigma2; % Derivative of square
   de_dsigma2 = sigma_range.*de_dsigma2; % derivative of 'unnormalization'
   de_dsigma2 = de_dsigma2 .* sigma2_der; % Derivative of thresholding
   
   % Sum
   de_dsigma1 = mean(de_dsigma1,3);
   de_dsigma2 = mean(de_dsigma2,3);
   
   % Alpha
   de_dalpha = de_dcol .* color_response2;
   de_dalpha = alpha_range.*de_dalpha; % derivative of 'unnormalization'
   de_dalpha = de_dalpha .* alpha_der; % Derivative of thresholding
   
   % Sum
   de_dalpha = mean(de_dalpha, 3);
end


end
