classdef unit
  %UNIT unit test for maxpool3d
  %   Detailed explanation goes here
  
  properties
    hdg; % handle to data generator
  end
  
  methods
    
    function ob = unit (hdg_)
      ob.hdg = hdg_;
    end
    
    function run (ob)
      print(ob);
        
      fprintf('verifying max index...\n');
      test_ind(ob);
      fprintf('done.\n');
      
%       fprintf('verifying dX...\n')
%       test_dX(ob);
%       fprintf('done.\n');
      
      fprintf('\n');
    end
    
    function print(ob)
      fprintf('description: %s\n', ob.hdg.desc);
      
      szX = size5d( ob.hdg.X );
      fprintf('X: [%d %d %d %d %d]\n', szX);
      
      if (~isempty(ob.hdg.pool))
        fprintf('%s\n', fmt_opt('pool', ob.hdg.pool) );
      end
      
      if (~isempty(ob.hdg.stride))
        fprintf('%s\n', fmt_opt('stride', ob.hdg.stride) );
      end
      
      if (~isempty(ob.hdg.pad) )
        fprintf('%s\n', fmt_opt('pad', ob.hdg.pad) );
      end
      
    end
    
    function test_dX (ob)
      % for convenient typing
      [X, pool, stride, pad, ran, ep] = get_all(ob);
      fprintf('range = %5.4f, ep = %5.4f\n', ran, ep);
      
      opts = {'pool',pool, 'stride',stride, 'pad',pad};
      % fprop
      [Y,ind] = mex_maxpool3d(X, opts{:});
      fprintf('Y: [%d %d %d %d %d]\n', size5d(Y));
      % bprop: fabricate a loss of sum
      tau  = 1;
      %dzdY = ones(size(Y), 'single'); 
      dzdY = ones(size(Y), 'like',Y);
      dzdX = mex_maxpool3d(dzdY,ind, opts{:});
      
      % grad chk
      function yy = fun_prop(arg)
        [yy,~] = mex_maxpool3d(arg, opts{:});
      end
      vl_testder(@fun_prop, X, dzdY, dzdX, ran*ep, tau);
    end
    
    function test_ind (ob)
      % for convenient typing
      [X, pool, stride, pad] = get_all(ob);
      
      % 
     	fprintf('fprop: generating Y and max index...\n');
      [Y, ind] = mex_maxpool3d(X,...
        'pool',pool, 'stride',stride, 'pad',pad);
      fprintf('Y: [%d %d %d %d %d]\n', size5d(Y) );
      % 
      fprintf('assert X(ind(:)) == Y\n');
      tmp = X(ind(:));
      assert(all( gather(tmp(:)) == gather(Y(:)) ));
      % 
      fprintf('bprop: generating all one dzdy, calculating dzdx...\n');
      dzdy = ones(size(Y),'like',Y);
      dzdx = mex_maxpool3d(dzdy,ind,size(X),...
        'pool',pool, 'stride',stride, 'pad',pad);
      fprintf('dzdX: [%d %d %d %d %d]\n', size5d(dzdx));
      %
      fprintf('assert dzdx be consistent with max index...\n');
      aa = zeros(size(dzdx), 'single'); % TODO: like
      for i = 1 : numel( ind )
        ix = ind(i);
        aa(ix) = aa(ix) + 1;
      end
      assert(all( gather(aa(:)) == gather(dzdx(:)) ));

    end % test_ind

  end % methods
end % unit  

function [X, pool, stride, pad, ran, ep] = get_all (ob)
  X = ob.hdg.X;
  pool   = ob.hdg.pool;
  stride = ob.hdg.stride;
  pad    = ob.hdg.pad;
  
  ep  = ob.hdg.ep;
  ran = ob.hdg.ran;
end % test_dX

