%% data
sz = [8,8,8, 5,9];
x = rand(sz, 'single');
% fprop
[y, ind] = mex_maxpool3d(x);
% bprop (in ConvNet it should be dy, here we just use y for illustration)
xx = mex_maxpool3d(y,ind);
%% validate
ix  = 5:6;
iy = 3;
yvalue = y(iy,iy,iy, 2,5);
subx = x(ix,ix,ix, 2,5);
subxx = xx(ix,ix,ix, 2,5);
%
assert( yvalue == max(subx(:)) )
assert( yvalue == max(subxx(:)) )
[~, im_x]  = max( subx(:) );
[~, im_xx] = max( subxx(:) );
assert( im_x == im_xx )

disp(yvalue)