function timeit_mp3d_bprop_gpu()
N = 256;
aa = 32;
szX = [aa,aa,1, 20, N];
pool = [2, 2, 1];
pad = [0,0, 0,0, 0,0];
stride = [1, 1, 1];
X = gpuArray.randn(szX, 'single');
T = 5;

te1 = time_mp3d();
fprintf('\n');
te2 = time_mp2d();

fprintf('mp3d: %5.4f\n', te1/T);
fprintf('vl: %5.4f\n', te2/T);
fprintf('mp3d/vl: %6.4f\n', te1/te2);

function te = time_mp3d()
  [Y,ind] = mex_maxpool3d(X,...
    'pool', pool,...
    'pad', pad,...
    'stride', stride);
  dY = rand(size(Y),'like',Y);

  te = tic;
  for t = 1 : T
    dX = mex_maxpool3d(dY, ind,...
      'pool',pool,...
      'pad', pad,...
      'stride', stride);
  end
  te = toc(te);
end

function te = time_mp2d()
  XX = squeeze(X);
  pool2d = pool(1:2);
  pad2d = pad(1:4);
  stride2d = stride(1:2);
  
  Y = vl_nnpool(XX, pool2d,...
    'pad', pad2d,...
    'stride', stride2d);
  dY = rand(size(Y), 'like',Y);

  te = tic;
  for t = 1 : T
  dXX = vl_nnpool(XX, pool2d, dY,...
    'pad', pad2d,...
    'stride', stride2d);
  end
  te = toc(te);
end




end

