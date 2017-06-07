%% data
szX = [8,8,8, 5,9];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];
X = gpuArray.rand(szX, 'single');
F = gpuArray.rand(szF, 'single');
B = gpuArray.rand(szB, 'single');
%% fprop
Y = mex_conv3d(X,F,B);
%% bprop
dZdY = rand(size(Y), 'like',Y);
[dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY);