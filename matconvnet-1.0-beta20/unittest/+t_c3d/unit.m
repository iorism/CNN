classdef unit
  %UNIT unit test for conv3d
  %   Detailed explanation goes here
  
  properties
    hdg; % handle to data generator
    ratio; % num elements to test
  end
  
  methods
    
    function ob = unit (hdg_)
      ob.hdg = hdg_;
      ob.ratio = 0.25;
    end
    
    function run (ob)
      print(ob);
     
      fprintf('verifying dX dF dB...\n');
      test_dXdFdB(ob);
      
      fprintf('done.\n\n');
    end
    
    function print(ob)
      fprintf('description: %s\n', ob.hdg.desc);
      
      szX = size5d( ob.hdg.X );
      fprintf('X: [%d %d %d %d %d]\n', szX);
      
      szF = size5d( ob.hdg.F );
      fprintf('F: [%d %d %d %d %d]\n', szF);
      
      szB = [1, size(ob.hdg.B,2)];
      fprintf('B: [%d %d]\n', szB);
      
      if (~isempty(ob.hdg.stride))
        fprintf('%s\n', fmt_opt('stride', ob.hdg.stride) );
      end
      
      if (~isempty(ob.hdg.pad) )
        fprintf('%s\n', fmt_opt('pad', ob.hdg.pad) );
      end
      
    end

    function test_dXdFdB (ob)
      % for convenient typing
      [X, F, B, stride, pad, ran, ep] = get_all(ob);
      fprintf('range = %5.4f, ep = %5.4f\n', ran, ep);
      
      opts = {'stride',stride, 'pad',pad};
      % fprop
      Y = mex_conv3d(X,F,B, opts{:});
      fprintf('Y: [%d %d %d %d %d]\n', size5d(Y));
      % bprop
      dzdY = rand(size(Y), 'like', Y);
      %dzdY = ones(size(Y), 'like', Y);
      [dzdX,dzdF,dzdB] = mex_conv3d(X,F,B, dzdY, opts{:});
      
      % grad chk
      vl_testder(... dX
        @(arg) mex_conv3d(arg,F,B, opts{:}),...
        X, dzdY, dzdX, ran*ep);
      vl_testder(... dF
        @(arg) mex_conv3d(X,arg,B, opts{:}),...
        F, dzdY, dzdF, ran*ep);
      vl_testder(... dB
        @(arg) mex_conv3d(X,F,arg, opts{:}),...
        B, dzdY, dzdB, ran*ep);
  
    end

  end % methods
end % unit  

function [X, F, B, stride, pad, ran, ep] = get_all (ob)
  X = ob.hdg.X;
  F = ob.hdg.F;
  B = ob.hdg.B;
  stride = ob.hdg.stride;
  pad    = ob.hdg.pad;
  
  ran = ob.hdg.ran;
  ep  = ob.hdg.ep;
end % test_dX

