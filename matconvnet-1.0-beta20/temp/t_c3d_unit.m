classdef unit
  %UNIT unit test for conv3d
  %   Detailed explanation goes here
  
  properties
    hdg; % handle to data generator
    num; % num elements to test
  end
  
  methods
    
    function ob = unit (hdg_)
      ob.hdg = hdg_;
      ob.num = 10;
    end
    
    function run (ob)
      print(ob);
      
      fprintf('verifying fprop: Y...\n');
      test_Y(ob);
      
      fprintf('verifying bprop: dX...\n');
      [r,s] = test_dX(ob);
      %fprintf('whether succeed: %d\n', all(s) );
      fprintf('max relative diff: %7.6f %%\n', 100*max(r) );
      
      fprintf('verifying bprop: dF...\n');
      [r,s] = test_dF(ob);
      %fprintf('whether succeed: %d\n', all(s) );
      fprintf('max relative diff: %7.6f %%\n', 100*max(r) );      
      
      fprintf('verifying bprop: dB...\n');
      [r,s] = test_dB(ob);
      %fprintf('whether succeed: %d\n', all(s) );
      fprintf('max relative diff: %7.6f %%\n', 100*max(r) );
      
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
    
    function test_Y (ob)
      
    end
    
    function [r, s] = test_dX (ob)
      % for convenient typing
      [X, F, B, stride, pad, ran, ep] = get_all(ob);
      
      % do the job
      ind = randperm( numel(X) );
      numnum = min(ob.num, numel(X));
      ind = [1, ind(1:numnum), numel(X)];
      for i = 1 : numel(ind)
        ii = ind(i);
        % by numeric loss difference
        szX = size(X);
        deltaX = zeros(szX, 'single'); % TODO: like
        deltaX(ii) = ep .* ran;
        %
        Y2 = mex_conv3d( X + deltaX, F, B, 'stride',stride, 'pad',pad);
        z2 = sum( Y2(:) );
        %
        Y1 = mex_conv3d( X - deltaX, F, B,  'stride',stride, 'pad',pad);
        z1 = sum( Y1(:) );
        %
        dzdep_app = (z2 - z1)/(2*ep*ran);
        
        % by calculation
        Y = mex_conv3d(X,F,B, 'stride',stride, 'pad',pad);
        dzdy = ones(size(Y), 'single');
        [dzdx,~,~] = mex_conv3d(X,F,B, dzdy,  'stride',stride, 'pad',pad);
        dzdep = dzdx(ii);
        
        % compare: succeed if < tolerance?
        tol = auto_tol(dzdep_app, dzdep);
        di = abs(dzdep_app - dzdep);
        s(i) = all( di < tol );
        % compare: relative difference
        r(i) = di/abs(dzdep+eps); %#ok<AGROW>
      end
     
    end
    
    function [r, s] = test_dF (ob)

      % for convenient typing
      [X, F, B, stride, pad, ran, ep] = get_all(ob);
      
      % do the job
      ind = randperm( numel(F) );
      numnum = min(ob.num, numel(F));
      ind = [1, ind(1:numnum), numel(F)];
      for i = 1 : numel(ind)
        ii = ind(i);
        % by numeric loss difference
        szF = size(F);
        deltaF = zeros(szF, 'single'); % TODO: like
        deltaF(ii) = ep .* ran;
        %
        Y2 = mex_conv3d( X, F+deltaF, B, 'stride',stride, 'pad',pad);
        z2 = sum( Y2(:) );
        %
        Y1 = mex_conv3d( X, F-deltaF, B,  'stride',stride, 'pad',pad);
        z1 = sum( Y1(:) );
        %
        dzdep_app = (z2 - z1)/(2*ep*ran);
        
        % by calculation
        Y = mex_conv3d(X,F,B, 'stride',stride, 'pad',pad);
        dzdy = ones(size(Y), 'single');
        [~,dzdF,~] = mex_conv3d(X,F,B, dzdy,  'stride',stride, 'pad',pad);
        dzdep = dzdF(ii);
        
        % compare: succeed if < tolerance?
        tol = auto_tol(dzdep_app, dzdep);
        di = abs(dzdep_app - dzdep);
        s(i) = all( di < tol );
        % compare: relative difference
        r(i) = di/abs(dzdep+eps); %#ok<AGROW>
      end
    end
    
    function [r, s] = test_dB (ob)
       % for convenient typing
      [X, F, B, stride, pad, ran, ep] = get_all(ob);
      
      % do the job
      ind = randperm( numel(B) );
      numnum = min(ob.num, numel(B));
      ind = [1, ind(1:numnum), numel(B)];
      for i = 1 : numel(ind)
        ii = ind(i);
        % by numeric loss difference
        szB = size(B);
        deltaB = zeros(szB, 'single'); % TODO: like
        deltaB(ii) = ep .* ran;
        %
        Y2 = mex_conv3d( X, F, B+deltaB, 'stride',stride, 'pad',pad);
        z2 = sum( Y2(:) );
        %
        Y1 = mex_conv3d( X, F, B-deltaB,  'stride',stride, 'pad',pad);
        z1 = sum( Y1(:) );
        %
        dzdep_app = (z2 - z1)/(2*ep*ran);
        
        % by calculation
        Y = mex_conv3d(X,F,B, 'stride',stride, 'pad',pad);
        dzdy = ones(size(Y), 'single');
        [~,~,dzdB] = mex_conv3d(X,F,B, dzdy,  'stride',stride, 'pad',pad);
        dzdep = dzdB(ii);
        
        % compare: succeed if < tolerance?
        tol = auto_tol(dzdep_app, dzdep);
        di = abs(dzdep_app - dzdep);
        s(i) = all( di < tol );
        % compare: relative difference
        r(i) = di/abs(dzdep+eps); %#ok<AGROW>
      end
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

