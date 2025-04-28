classdef psf_phi0_LRDOE < dagnn.Layer
    % Create depth dependent psf
    %
    % Forward:
    % P = psf(d,phi0);
    %
    % phi0 = nominal channel phase mask (parameter)
    % d = depth (input)
    %
    % Formula:
    % P = abs(F(A*exp(j*phi)*exp(def(d)))).^2, where def(d) is the depth
    %                                          dependent defocus, and F(.)
    %                                          is the Fourier transform
    % Steps:
    % Create PSF
    % Normalize
    % Downsample to Sensor Grid
    %
    % Backward:
    % [dedd, dedphi0] = psf(d,phi0,dedP);
    %
    % Ugur Akpinar, TUT 2018
    
    properties
        opts = struct();
        h_std = [0e-9 0e-9];
    end
    
    methods
        function outputSizes = getOutputSizes(obj, inputSizes)
            N = inputSizes{1};
            b = obj.opts(1).baseKernelSize;
            nch = obj.opts(1).numchannel;
            outputSizes = cell(1);
            outputSizes{1} = [b b nch N(4)];
        end
        
        function outputs = forward(obj, inputs, params)
            
            % input
            d = inputs{1};
            % parameter
            phi0_lr = params{1};
            
            if isfield(obj.opts,'U1x')
               % Upsample phi0
               phi0_lr = phi0_lr .* obj.opts(1).M_doe;
               U1x = obj.opts(1).U1x;
               U1y = obj.opts(1).U1y;
               phi0 = U1x*phi0_lr*U1y';
            else
               phi0 = phi0_lr; 
            end
            
            % Nominal channel
            ch0 = obj.opts(1).ch0;
            
            % Add random noise to phase
            n0 = obj.opts(ch0).ref_idx;
            lambda0 = obj.opts(ch0).lambda;
            min_std = obj.h_std(1);
            max_std = obj.h_std(2);
            n_std = min_std + (max_std-min_std)*rand(1);
            noise_std = 2*pi*(n0-1)*n_std/lambda0;
            if noise_std>0
                noise = TruncatedGaussian(noise_std, [-2*noise_std 2*noise_std], size(phi0), class(phi0));
                phi0 = phi0 + noise;
            end
            % Mask the phase outside the aperture
            phi0 = phi0 .* obj.opts(1).M;
            
            % gpu
            isgpu = isa(phi0, 'gpuArray') || isa(d, 'gpuArray');
            if (isgpu)
                d = gpuArray(d);
                phi0 = gpuArray(phi0);
            end
            
            % single
            d = single(d);
            phi0 = single(phi0);
            
            % Derive phase for each color
            phi = phi0_to_phi_MS(obj.opts, phi0);

            % output
            psf = obj.forwardCore(d,phi);
            outputs{1} = psf;
            
        end
        
        function psfd = forwardCore(obj, d, phi)
            
            % Forward:
            % Calculate psf at the sensor grid
            % Derive psf analitically
            % Normailze s.t. the sum is 1
            % Downsample to sensor pixel size
            
            % Camera parameters
            cam = obj.opts;
            
            % Amplitude mask
            A = cam(1).M;
            
            % obtain psf from code and depth
            psf = createpsf(cam, d, A, phi);
            
            % normalization
            psfn = normalizepsf(cam, psf);
            
            % Resampling
            b = cam(1).baseKernelSize;
            nch = cam(1).numchannel;
            psfd = zeros(b,b,nch,size(psfn,4),'like',psfn);
            for ch = 1:nch
                D1 = cam(ch).D1;
                psfd1 = tensor_mtx_mult(psfn(:,:,ch,:),D1,1);
                psfd(:,:,ch,:) = tensor_mtx_mult(psfd1,D1,2);
            end
                        
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            % input
            d = inputs{1};
            % parameter
            phi0_lr = params{1};
            
            if isfield(obj.opts,'U1x')
               % Upsample phi0
               phi0_lr = phi0_lr .* obj.opts(1).M_doe;
               U1x = obj.opts(1).U1x;
               U1y = obj.opts(1).U1y;
               phi0 = U1x*phi0_lr*U1y';
            else
               phi0 = phi0_lr; 
            end
            
            phi0 = phi0 .* obj.opts(1).M;
            
            % output derivative
            de_dpsf = derOutputs{1}; 

            % gpu
            isgpu = isa(phi0,'gpuArray') || isa(d,'gpuArray') || isa(de_dpsf,'gpuArray');
            if (isgpu)
                d = gpuArray(d);
                phi0 = gpuArray(phi0);
                de_dpsf = gpuArray(de_dpsf);
            end
            
            % single
            d = single(d);
            phi0 = single(phi0);
            de_dpsf = single(de_dpsf);
                        
            % Derive phase for each color
            phi = phi0_to_phi_MS(obj.opts, phi0);
                 
            de_dphi = obj.backwardCore(de_dpsf, d, phi);
                        
            % Derive phase for each color
            [~,de_dphi0] = phi0_to_phi_MS(obj.opts,phi0,de_dphi);
            
            de_dphi0 = de_dphi0 .* obj.opts(1).M;
            
            if isfield(obj.opts,'U1x')
               % Downsample back
                de_dphi0_lr = U1x'*de_dphi0*U1y;
                de_dphi0_lr = de_dphi0_lr .* obj.opts(1).M_doe;
            else
               de_dphi0_lr = de_dphi0; 
            end
            
            derParams = cell(size(params));
            derParams{1} = de_dphi0_lr;

            % store
            derInputs{1} = single(0);
            if isgpu
                derParams = cellfun(@(p)gpuArray(p),derParams,'UniformOutput',false);
                derInputs = cellfun(@(p)gpuArray(p),derInputs,'UniformOutput',false);
            end
            
        end
        
        function  de_dphi = backwardCore(obj, de_dpsfd, d, phi)
            
            % Backward
            % Upsample the psf derivative
            % 'Unnormalize'
            % Calculate derivatives wrt amplitude and phase
            
            % Camera parameters
            cam = obj.opts;
            
            % Amplitude mask
            A = cam(1).M;
            
            % Resampling
            b = size(A);
            nch = cam(1).numchannel;
            de_dpsfn = zeros(b(1),b(2),nch,size(de_dpsfd,4),'like',de_dpsfd);
            for ch = 1:nch
                D1 = cam(ch).D1;
                de_dpsfd1 = tensor_mtx_mult(de_dpsfd(:,:,ch,:),D1',1);
                de_dpsfn(:,:,ch,:) = tensor_mtx_mult(de_dpsfd1,D1',2);
            end
            
            % Unnormalize
            [~, de_dpsf] = normalizepsf(cam, [], de_dpsfn);
            
            % Obtain derivates wrt normalized masks
            [~, ~, de_dphi] = createpsf(cam, d, A, phi, de_dpsf);
            
        end
        
        function params = initParams(obj)
            if isfield(obj.opts,'phi0init')
                phi0 = obj.opts(1).phi0init;
            elseif isfield(obj.opts,'Zern')
                coef = obj.opts(1).coef;
                % Zernike polynomials
                Zern = obj.opts(1).Zern;
                % Nominal phase
                phi0 = Zern_to_phi0(Zern,coef);
            elseif isfield(obj.opts,'p_x_doe')
                phi0 = zeros(size(obj.opts(1).p_x_doe), 'single');
            else
                phi0 = zeros(size(obj.opts(1).p_x), 'single');                
            end
            
            if isfield(obj.opts,'M_doe')
                phi0 = phi0.* obj.opts(1).M_doe;
            else
                phi0 = phi0.* obj.opts(1).M;
            end
            params{1} = phi0;
        end
        
        function obj = psf_phi0_LRDOE(varargin)
            obj.load(varargin);
        end
    end
end