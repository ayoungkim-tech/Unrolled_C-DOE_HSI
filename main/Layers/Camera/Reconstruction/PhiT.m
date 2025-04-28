classdef PhiT < dagnn.Layer
    % Transpose of the measurement matrix
    % I_HS = PhiT(I,PSF,ColorFilt)
    
    properties
        opts = struct();
        sc = 1;
    end
    
    methods
        function outputSizes = getOutputSizes(obj, inputSizes)
            sz = inputSizes{1};
            nch = inputSizes{2}(3);
            sz(3) = nch;
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
            
            % batch
            assert(size(I,4) == size(psf,4));
            
            % Sensor image to HS cube
            I_HS = colT(I,color_response);
            
            % scale
            I_HS = I_HS .* obj.sc;
            
            % PSF Transpose
            I_HS = obj.forwardCore(I_HS, psf);
            
            % output
            outputs{1} = I_HS;
            
            if (isgpu)
                outputs{1} = gpuArray(outputs{1});
            end
        end
        
        function I_HS = forwardCore(obj, I, psf)
            
            % PSF transpose
            H = @(a,f) real(fftconv(single(a),single(f),'same')); % Note that during sensor operation, the psf was rotated, so we don't rotate here.
            % convolve
            I_HS = H(I,psf);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            % input
            I = inputs{1};
            psf = inputs{2};
            color_response = inputs{3};
            
            % Output derivative
            de_dI_HS = derOutputs{1};
            
            isgpu = isa(I, 'gpuArray') || isa(psf, 'gpuArray');
            if (isgpu)
                I = gpuArray(single(I));
                psf = gpuArray(single(psf));
                color_response = gpuArray(single(color_response));
                de_dI_HS = gpuArray(single(de_dI_HS));
            else
                I = single(I);
                psf = single(psf);
                color_response = single(color_response);
                de_dI_HS = single(de_dI_HS);
            end
            
            % First, Sensor image to HS cube again
            I_HS = colT(I,color_response);
            
            % scale
            I_HS = I_HS .* obj.sc;
            
            % Calculate PSF Derivative and update de_dI_HS
            [de_dI_HS, de_dpsf] = obj.backwardCore(I_HS, de_dI_HS, psf);
            
            % Derivative of scale
            de_dI_HS = de_dI_HS .* obj.sc;
            
            % Back to de_dI
            [~,de_dI,dedcol] = colT(I,color_response,de_dI_HS);
            
            derInputs{1} = de_dI;
            derInputs{2} = de_dpsf;
            derInputs{3} = dedcol;
            
            derParams{1} = [];
            
            if (isgpu)
                derInputs = cellfun(@(x) gpuArray(x), derInputs, 'UniformOutput', false);
                derParams = cellfun(@(x) gpuArray(x), derParams, 'UniformOutput', false);
            end
            
        end
        
        function [de_dI, de_dpsf] = backwardCore(obj, I, de_dI_HS, psf)
            
            psfSize = size(psf,1);
            
            if mod(psfSize(1),2)
                % Check here
                Ip = padarray(I,[floor(psfSize/2) floor(psfSize/2)],0,'both');
                de_dpsf = real(fftconv(Ip(end:-1:1,end:-1:1,:,:),de_dI_HS,'valid'));
                de_dI = real(fftconv(de_dI_HS,psf(end:-1:1,end:-1:1,:,:),'same'));
            else
                % Check here
                [de_dI, de_dpsf] = vl_nnconv(I, psf, [], de_dI_HS, 'pad', (psfSize)/2);
            end
            
        end
        
        function params = initParams(obj)
            params{1} = [];
        end
        
        function obj = PhiT(varargin)
            obj.load(varargin);
        end
    end
end