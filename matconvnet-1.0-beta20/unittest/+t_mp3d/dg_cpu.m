classdef dg_cpu
  %DG_CPU Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    X;
    pool;
    stride;
    pad;
    
    desc;
    ran; % numeric range
    ep;  % epsilon
  end
  
  methods
    function ob = dg_cpu(szX, pool, stride, pad, desc)
      
      ob.ep  = 1e-2;
      ob.ran = 100;
      
      ob.X = ob.ran * randn(szX, 'single');
      
      ob.pool = [];
      if (~isempty(pool))
        ob.pool = pool;
      end
      
      ob.stride = [];
      if (~isempty(stride))
        ob.stride = stride;
      end
      
      ob.pad = [];
      if (~isempty(pad))
        ob.pad = pad;
      end
      
      ob.desc = ['mp3d, cpu array, ', desc];
    end % dg_cpu
    
  end % methods
  
end

