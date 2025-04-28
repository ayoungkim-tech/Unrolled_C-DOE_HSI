classdef PReLU < dagnn.ElementWise
  properties
    useShortCircuit = true
    opts = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnrelu(inputs{1}, [], ...
                             'leak', params{1}, obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnrelu(inputs{1}, derOutputs{1}, ...
                               'leak', params{1}, ...
                               obj.opts{:}) ;
      da =  derOutputs{1}.*min(0,inputs{1});                     
      derParams{1} = sum(da(:));
    end

    function forwardAdvanced(obj, layer)
      if ~obj.useShortCircuit || ~obj.net.conserveMemory
        forwardAdvanced@dagnn.Layer(obj, layer) ;
        return ;
      end
      net = obj.net ;
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      param = layer.paramIndexes ;
      
      net.vars(out).value = vl_nnrelu(net.vars(in).value, [], ...
                                      'leak', net.params(param).value, ...
                                      obj.opts{:}) ;
      net.numPendingVarRefs(in) = net.numPendingVarRefs(in) - 1;
      net.numPendingParamRefs(param) = net.numPendingParamRefs(param) - 1;
      
      if ~net.vars(in).precious & net.numPendingVarRefs(in) == 0
        net.vars(in).value = [] ;
      end    
    end

    function backwardAdvanced(obj, layer)
      if ~obj.useShortCircuit || ~obj.net.conserveMemory
        backwardAdvanced@dagnn.Layer(obj, layer) ;
        return ;
      end
      net = obj.net ;
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      param = layer.paramIndexes ;
      
      if isempty(net.vars(out).der), return ; end

      derInput = vl_nnrelu(net.vars(in).value, net.vars(out).der, ...
                           'leak', net.params(param).value, obj.opts{:}) ;
      
      da =  net.vars(out).der.*min(0,net.vars(in).value);                     
      derParam = sum(da(:));
      
      if ~net.vars(out).precious
        net.vars(out).der = [] ;
        net.vars(out).value = [] ;
      end

      if net.numPendingVarRefs(in) == 0
          net.vars(in).der = derInput ;
      else
          net.vars(in).der = net.vars(in).der + derInput ;
      end
      net.numPendingVarRefs(in) = net.numPendingVarRefs(in) + 1 ;
      
      if net.numPendingParamRefs(param) == 0
          net.params(param).der = derParam ;
      else
          net.params(param).der = net.params(param).der + derParam ;
      end
      net.numPendingParamRefs(param) = net.numPendingParamRefs(param) + 1 ;
          
    end
    
    function params = initParams(obj)
        params{1} = 0.25;        
    end

    function obj = PReLU(varargin)
      obj.load(varargin) ;
    end
  end
end
