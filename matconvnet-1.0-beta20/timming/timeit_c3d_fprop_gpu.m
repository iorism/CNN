function timeit_c3d_fprop_gpu()
N = 256;
aa = 36;
bb = 5;
szX = [aa,aa,1, 20, N];
szF = [bb, bb, 1, 20, 16];
szB = [1,16];
pad = [0,0, 0,0, 0,0];
stride = [1, 1, 1];
X = gpuArray.randn(szX, 'single');
F = gpuArray.randn(szF, 'single');
B = gpuArray.randn(szB, 'single');
T = 5;

te1 = time_conv3d();
fprintf('\n');
te2 = time_conv2d();

fprintf('conv3d: %5.4f\n', te1/T);
fprintf('vl: %5.4f\n', te2/T);
fprintf('conv3d/vl: %6.4f\n', te1/te2);

function te = time_conv3d()
  Y1 = mex_conv3d(X,F,B,...
  'pad', pad,...
  'stride', stride);

  te = tic;
  for t = 1 : T
    Y1 = mex_conv3d(X,F,B,...
      'pad', pad,...
      'stride', stride);
  end
  te = toc(te);
end

function te = time_conv2d()
  XX = squeeze(X);
  FF = squeeze(F);
  pad2d = pad(1:4);
  stride2d = stride(1:2);
  
  Y1 = vl_nnconv(XX, FF, B,...
  'pad', pad2d,...
  'stride', stride2d);

  te = tic;
  for t = 1 : T
    Y1 = vl_nnconv(XX, FF, B,...
      'pad', pad2d,...
      'stride', stride2d);
  end
  te = toc(te);
end




end

