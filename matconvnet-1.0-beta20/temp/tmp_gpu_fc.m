szX = [2, 2, 2, 1, 1];
szF = [2, 2, 2, 1, 2];
szB = [1, 2];
X = gpuArray.ones(szX, 'single');
F = gpuArray.ones(szF, 'single');
B = gpuArray.zeros(szB, 'single');

Y = mex_conv3d(X,F,B);