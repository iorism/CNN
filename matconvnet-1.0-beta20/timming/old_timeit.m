function timeit ()
%% data
szX = [28,28, 1, 20, 256];
szF = [5,5, 1, 20,16];
szB = [1, szF(end)];
X = gpuArray.rand(szX, 'single');
F = gpuArray.rand(szF, 'single');
B = gpuArray.rand(szB, 'single');

  function time_conv3d()
    Y = conv
  end


end

%% fprop
Y = mex_conv3d(X,F,B);
%% bprop
% dZdY = rand(size(Y), 'like',Y);
% [dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY);