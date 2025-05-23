classdef Subtract < dagnn.ElementWise
  %Subtract DagNN Subtract layer
  %   The Subtract layer subtracts the second input from the first and store the result
  %   as its only output.

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = inputs{1} - inputs{2};
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = derOutputs{1} ;
      derInputs{2} = -1*derOutputs{1} ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Subtract layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

%     function rfs = getReceptiveFields(obj)
%       numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
%       rfs.size = [1 1] ;
%       rfs.stride = [1 1] ;
%       rfs.offset = [1 1] ;
%       rfs = repmat(rfs, numInputs, 1) ;
%     end

    function obj = Subtract(varargin)
      obj.load(varargin) ;
    end
  end
end
