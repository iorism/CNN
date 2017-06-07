function conv3d_vs_matconvnet_fprop_gpu()

rng(6345, 'twister');

disp('dim1, dim2 as 2d conv')
szX = [1,1, 1, 1, 1];
szF = [2,2, 1, 1, 1];
szB = [1,1];
pad = [1,0, 1,0, 0,0];
stride = [1, 1, 1];
% szX = [13,10,1, 2, 5];
% szF = [3, 5, 1, 2, 4];
% szB = [1,4];
% pad = [0,0, 0,0, 0,0];
% stride = [2, 3, 1];

cmp_3d2d(szX, szF, szB,...
  pad, stride,...
  pad(1:4), stride(1:2) )


disp('dim1, dim3 as 2d conv')
% szX = [13, 1, 10, 2, 5];
% szF = [3,  1, 5,  2, 4];
% szB = [1, 4];
% pad = [0,2, 0,0, 1,5];
% stride = [2, 1, 3];
szX = [13, 1, 10, 2, 5];
szF = [3,  1, 5,  2, 4];
szB = [1, 4];
pad = [0,0, 0,0, 0,0];
stride = [2, 1, 3];

cmp_3d2d(szX, szF, szB,...
  pad, stride,...
  [pad(1:2),pad(5:6)], [stride(1),stride(3)] );


disp('dim2, dim3 as 2d conv')
% szX = [1, 13,  10, 2, 5];
% szF = [1, 3,   5,  2, 4];
% szB = [1, 4];
% pad = [0,0, 0,2, 1,5];
% stride = [1, 2, 3];
szX = [1, 13,  10, 2, 5];
szF = [1, 3,   5,  2, 4];
szB = [1, 4];
pad = [0,0, 0,0, 0,0];
stride = [1, 2, 3];

cmp_3d2d(szX, szF, szB,...
  pad, stride,...
  pad(3:6), stride(2:3) );

%%%
% szX = [3,13,1, 4, 5];
% szF = [2,2, 1, 4, 7];
% szB = [1,7];
% pad = [2,1, 3,2, 0,0];
% stride = [2, 1, 1];

%%%
% szX = [7, 7, 1, 3, 16];
% szF = [3, 4, 1, 3, 5];
% szB = [1,5];
% pad = [2,1, 1,2, 0,0];
% stride = [2, 1, 1];


function cmp_3d2d(szX, szF, szB, pad3d, stride3d, pad2d, stride2d)

ran = 50;

% gen data
X = ran * gpuArray.randn(szX, 'single');
F = ran * gpuArray.randn(szF, 'single');
B = ran * gpuArray.randn(szB, 'single');

% 3d conv
Y1 = mex_conv3d(X,F,B,...
  'pad', pad3d,...
  'stride', stride3d);

% 2d conv: matconvnet
Y2 = vl_nnconv( squeeze(X), squeeze(F), B,...
  'pad', pad2d,...
  'stride', stride2d );

% the size should be same (except for the singular dim 3)
disp('size(Y1) as 3d conv:')
disp(size(Y1))
disp('size(Y2) as 2d conv:')
disp(size(Y2))

% the values should be same
diff_Y = abs(Y1(:)-Y2(:));
tau = 1e-8; 
fprintf('assert Y1 - Y2 is very small\n\n\n');
assert( all( gather(diff_Y) < gather(tau) ) );