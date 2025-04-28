classdef Phi < dagnn.Layer
    % Measurement model
    % I = Phi(I_HS,PSF,ColorFilt)
    
    properties
        opts = struct();
        s_std = [0.001 0.015];
    end
    
    methods
        function outputSizes = getOutputSizes(obj, inputSizes)
            sz = inputSizes{1};
            sz(3) = 1;
            outputSizes{1} = sz;
        end
        
        function outputs = forward(obj, inputs, params)
            
            % input
            I = inputs{1};
            psf = inputs{2};
            color_response = inputs{3};
            
            isgpu = isa(I, 'gpuArray') || isa(psf, 'gpuArray');
            if (isgpu)
                I = gpuArray(single(I));
                psf = gpuArray(single(psf));
                color_response = gpuArray(single(color_response));
            else
                I = single(I);
                psf = single(psf);
                color_response = single(color_response);
            end
            
            % channel
            assert(size(I,3) == size(psf,3));
            % batch
            assert(size(I,4) == size(psf,4));
            
            % Sensor image
            Ib = obj.forwardCore(I, psf);
            
            % Sensor spectral response to resulting image
            Ib = Sensor_image(Ib,color_response);
            
            % Sensor NoiseConv
            max_std = obj.s_std(1);
            min_std = obj.s_std(2);
            std = min_std + (max_std - min_std)*rand(1);
            if std>0
                noise = TruncatedGaussian(std, [-2*std 2*std], size(Ib), class(Ib));
                Ib = Ib + noise;
            end
            
            % output
            outputs{1} = Ib;
            
            if (isgpu)
                outputs{1} = gpuArray(outputs{1});
            end
            
        end
        
        function Ib = forwardCore(obj, I, psf)
            % Convolution function
            H = @(a,f) real(fftconv(single(a),single(f(end:-1:1,end:-1:1,:,:)),'same'));
            % convolve
            Ib = H(I,psf);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            % input
            I = inputs{1};
            psf = inputs{2};
            color_response = inputs{3};
            
            % Output derivative
            de_dIb = derOutputs{1};
            
            isgpu = isa(I, 'gpuArray') || isa(psf, 'gpuArray');
            if (isgpu)
                I = gpuArray(single(I));
                psf = gpuArray(single(psf));
                color_response = gpuArray(single(color_response));
                de_dIb = gpuArray(single(de_dIb));
            else
                I = single(I);
                psf = single(psf);
                color_response = single(color_response);
                de_dIb = single(de_dIb);
            end
            
            % First, create the sensor image again
            Ib = obj.forwardCore(I, psf);
            
            % Derivative wrt the sensor repsonse parameters
            [~,de_dIb,dedcol] = Sensor_image(Ib,color_response,de_dIb);
            
            % Input Derivative 
            [de_dI, de_dpsf] = obj.backwardCore(I, de_dIb, real(psf));
            
            derInputs{1} = de_dI;
            derInputs{2} = de_dpsf;
            derInputs{3} = dedcol;
            
            derParams{1} = [];
            
            if (isgpu)
                derInputs = cellfun(@(x) gpuArray(x), derInputs, 'UniformOutput', false);
                derParams = cellfun(@(x) gpuArray(x), derParams, 'UniformOutput', false);
            end
            
        end
        
        function [de_dI, de_dpsf] = backwardCore(obj, I, de_dIb, psf)
            
            psfSize = size(psf,1);
            
            if mod(psfSize(1),2)
                % Check here
                Ip = padarray(I,[floor(psfSize/2) floor(psfSize/2)],0,'both');
                de_dpsf = real(fftconv(Ip,de_dIb(end:-1:1,end:-1:1,:,:),'valid'));
                de_dI = real(fftconv(de_dIb,psf,'same'));
            else
                % Check here
                [de_dI, de_dpsf] = vl_nnconv(I, psf, [], de_dIb, 'pad', (psfSize)/2);
            end
            
        end
        
        function params = initParams(obj)
            params{1} = [];
        end
        
        function obj = Phi(varargin)
            obj.load(varargin);
        end
    end
end