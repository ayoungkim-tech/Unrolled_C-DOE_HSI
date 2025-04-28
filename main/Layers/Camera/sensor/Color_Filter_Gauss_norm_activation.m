classdef Color_Filter_Gauss_norm_activation < dagnn.Layer
    % Create Color Filter from parameters
    
    properties
        opts = struct();
        norm = 'sum'; %'max', 'sum', or 'none'
        activation = 'sin'; %'sin', 'tanh' or 'clip'
    end
    
    methods
        function outputSizes = getOutputSizes(obj, inputSizes)
            nch = obj.opts(1).numchannel;
            fsize = obj.opts(1).colfiltersize;
            sz = [fsize,nch];
            outputSizes{1} = sz;
        end
        
        function outputs = forward(obj, inputs, params)
            
            % sensor color response
            mu = params{1};
            sigma = params{2};
            alpha = params{3};
            
            switch lower(obj.activation)
                case 'sin' % activation by sin: [0,1]
                    mu = 1/2 * (sin(mu)+1);
                    sigma = 1/2 * (sin(sigma)+1);
                    alpha = 1/2 * (sin(alpha)+1);
                case 'tanh'
                    mu = 1/2 * (tanh(mu)+1);
                    sigma = 1/2 * (tanh(sigma)+1);
                    alpha = 1/2 * (tanh(alpha)+1);     
                case 'clip' % clip
                    % Clip activation is done inside
                    % color_filter_double_Gauss_norm.m, no need to do it
                    % here again
                otherwise % error
                    error('Unknown activation %s. The options are: sin, tanh or clip.', obj.activation);
            end 
            
            mu1 = mu(:,:,1);
            mu2 = mu(:,:,2);
            sigma1 = sigma(:,:,1);
            sigma2 = sigma(:,:,2);
            
            color_response = color_filter_double_Gauss_norm(obj.opts,mu1,sigma1, mu2, sigma2,alpha);
            % Normalize
            switch lower(obj.norm)
                case {'sum','max'} % normalize by sum or by maximum of sum
                    color_response_n = normalize_ch_v2(color_response,obj.norm);
                case 'none' %no normalization
                    color_response_n = color_response;
                otherwise % error
                    error('Unknown normalization %s. The options are: none, max or sum.', obj.norm);
            end  
            
            % Show color response (with rgb colors)
            z = obj.opts(12).z_f;
            lambda_MS = [obj.opts.lambda]*1e+9;
            % use the D65 illuminant
            illuminant=65;
            % do minor thresholding
            threshold=0.001;
            % Plot color filter in rgb
            [ydim,xdim,zdim]=size(color_response_n);
            % reorder data so that each column holds the spectra of of one pixel
            Z = reshape(color_response_n,[],zdim);
            %Create the RBG image,
            col_rgb = HSI2RGB(lambda_MS,Z,ydim,xdim,illuminant,threshold);
            
            figure(2);
            subplot(222); imshow(col_rgb);
            title('color filter');
            
            % Plot gaussian components of each color in color response
%             figure(3);
%             lambda = cat(2,obj.opts.lambda).*1e+9;
%             color_response_2d =  reshape(color_response, [], obj.opts(1).numchannel); 
% %             color_response_sum = sum(color_response_2d, 1);
% %             plot(lambda, color_response_sum(:));
%             for idx = 1:size(color_response_2d,1)
%                 plot(lambda, color_response_2d(idx, :));
%                 hold on
%             end
%             hold off
%             xlim ([420 720])
%             title('Gaussian components of color filter')
            
            
            % output
            outputs{1} = color_response_n;

        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
                        
            % sensor color response
            mu = params{1};
            sigma = params{2};
            alpha = params{3};
            
            switch lower(obj.activation)
                case 'sin' % activation by sin: [0,1]
                    % Will be used for the backward pass
                    mu_der = 1/2 * cos(mu);
                    sigma_der = 1/2 * cos(sigma);
                    alpha_der = 1/2 * cos(alpha); 
                    
                    mu = 1/2 * (sin(mu)+1);
                    sigma = 1/2 * (sin(sigma)+1);
                    alpha = 1/2 * (sin(alpha)+1);
                case 'tanh'
                    % Will be used for the backward pass
                    mu_tanh = tanh(mu);
                    sigma_tanh = tanh(sigma);
                    alpha_tanh = tanh(alpha);
                    mu_der = 1/2 .* (1 - mu_tanh.^2);
                    sigma_der = 1/2 .* (1 - sigma_tanh.^2);
                    alpha_der = 1/2 .* (1 - alpha_tanh.^2);
                    
                    mu = 1/2 * (mu_tanh + 1);
                    sigma = 1/2 * (sigma_tanh + 1);
                    alpha = 1/2 * (alpha_tanh + 1);
                case 'clip' % clip
                    % Clip activation is done inside
                    % color_filter_double_Gauss_norm.m, no need to do it
                    % here again
                    % Will be used for the backward pass
                    mu_der = 0*mu + 1; % Just ones
                    sigma_der = 0*sigma + 1;
                    alpha_der = 0*alpha + 1;    
                otherwise % error
                    error('Unknown activation %s. The options are: sin or clip.', obj.activation);
            end  
            
            mu1 = mu(:,:,1);
            mu2 = mu(:,:,2);
            sigma1 = sigma(:,:,1);
            sigma2 = sigma(:,:,2);
            
            % Output derivative
            dedcoln = derOutputs{1};
            
            % First, calculate the color response again, from the
            % parameters
            color_response = color_filter_double_Gauss_norm(obj.opts,mu1,sigma1, mu2, sigma2,alpha);
            
            % Unnormalize
            switch lower(obj.norm)
                case {'sum','max'} % normalize by sum or by maximum of sum
                    [~,dedcol] = normalize_ch_v2(color_response,obj.norm,dedcoln);
                case 'none' %no normalization
                    dedcol = dedcoln;
                otherwise % error
                    error('Unknown normalization %s. The options are: none, max or sum.', obj.norm);
            end 
            
            % Change from color response to mu,sigma,alpha values
            [~,de_dmu1, de_dmu2, de_dsigma1, de_dsigma2, de_dalpha] = color_filter_double_Gauss_norm(obj.opts,mu1,sigma1,mu2,sigma2,alpha,dedcol);
            
            de_dmu = cat(3,de_dmu1,de_dmu2);
            de_dsigma = cat(3,de_dsigma1,de_dsigma2);
            
            % Derivative of the activation function
            de_dmu = de_dmu .* mu_der;
            de_dsigma = de_dsigma .* sigma_der;
            de_dalpha = de_dalpha .* alpha_der;
            
            % Parameter Derivatives
            derParams{1} = de_dmu;
            derParams{2} = de_dsigma;
            derParams{3} = de_dalpha;
            
            derInputs = cell(0);
            
        end
        
        function params = initParams(obj)
            fsize = obj.opts(1).colfiltersize;
            
            % Uniform random values between min and max
            mu = rand([fsize 2],'single');
            sigma = rand([fsize 2],'single');
            alpha = rand(fsize,'single');
            
            if strcmp(obj.activation,'sin')
                mu = 2*pi*mu;
                sigma = 2*pi*sigma;
                alpha = 2*pi*alpha;
            end
            
            if strcmp(obj.activation,'tanh')
                mu = randn([fsize 2],'single');
                sigma = randn([fsize 2],'single');
                alpha = randn(fsize,'single');
            end           
            
            params{1} = mu;
            params{2} = sigma;
            params{3} = alpha;
        end
        
        function obj = Color_Filter_Gauss_norm_activation(varargin)
            obj.load(varargin);
        end
    end
end