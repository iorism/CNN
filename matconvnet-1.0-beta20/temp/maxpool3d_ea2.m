%% data
sz = [9,9,8, 3,7];
x = rand(sz, 'single');
% fprop
[y, ind] = mex_maxpool3d(x, 'pool',[3,3,2]);
% bprop (in ConvNet it should be dy, here we just use y for illustration)
xx = mex_maxpool3d(y,ind, 'pool',[3,3,2]);
%% validate
ix12 = 4:6;
ix3  = 5:6;
iy12 = 2;
iy3  = 3;
yvalue = y(iy12,iy12,iy3, 2,5);
subx = x(ix12,ix12,ix3, 2,5);
subxx = xx(ix12,ix12,ix3, 2,5);
%
assert( yvalue == max(subx(:)) )
assert( yvalue == max(subxx(:)) )
[~, im_x]  = max( subx(:) );
[~, im_xx] = max( subxx(:) );
assert( im_x == im_xx )

yvalue
