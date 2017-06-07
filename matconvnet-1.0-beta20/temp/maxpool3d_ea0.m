%% data
sz = [8,8,8, 5,9];
x = rand(sz, 'single');
%% fprop
[y, ind] = mex_maxpool3d(x);
%% bprop 
dzdy = rand(size(y), 'single');
dzdx = mex_maxpool3d(dzdy,ind);