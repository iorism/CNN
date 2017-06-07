classdef dg_gpu
  %DG_GPU Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    X;
    F;
    B;
    stride;
    pad;
    
    desc;
    ran; % numeric range
    ep;  % epsilon
  end
  
  methods
    function ob = dg_gpu(szX, szF, szB, stride, pad, desc)
      
      ob.ep  = gpuArray(1e-2);
      ob.ran = gpuArray(100);
      
      ob.X = ob.ran * gpuArray.randn(szX, 'single');
      ob.F = ob.ran * gpuArray.randn(szF, 'single');
      ob.B = ob.ran * gpuArray.randn(szB, 'single');
      
      ob.stride = [];
      if (~isempty(stride))
        ob.stride = stride;
      end
      
      ob.pad = [];
      if (~isempty(pad))
        ob.pad = pad;
      end
      
      ob.desc = ['conv3d, gpu array ', desc];
    end % dg_cpu
    
  end % methods
  
end

