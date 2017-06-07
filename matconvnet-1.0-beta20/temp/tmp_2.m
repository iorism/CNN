%% data
sz = [32,32, 32*10,9];
x = gpuArray.rand(sz, 'single');
%% fprop
y = vl_nnpool(x, 2);
%% bprop 
% dzdy = gpuArray.ones(size(y), 'single');
% dzdx = mex_maxpool3d(dzdy,ind);
%%
% a1 = x( dzdx > 0 );
% a2 = x( sort(ind(:),'ascend') );
% a1 = gather(a1);
% a2 = gather(a2);
% assert( all( a1(:) == a2(:) ) );
%%
clear all