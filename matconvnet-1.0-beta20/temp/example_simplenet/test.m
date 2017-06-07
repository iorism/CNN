%% Config
K = 6;
N = 1;
%% Init
X = 1e-4 * gpuArray.rand(60,80,13,1,N,'single');   % random data
label = reshape([0,0,1,0,0,0], K, N);

pool  = [2,2,2];          % 3D pooling window size
p_str = [2,2,1];           % 3D pooling stride 
c_str = [1,1,1];           % 3D conv stride

F1 = 0.01*(gpuArray.rand(7,7,3,1,100, 'single')); B1 = 0.01*(gpuArray.rand(1,100, 'single'));
F3 = 0.01*(gpuArray.rand(5,5,3,100,50,'single')); B3 = 0.01*(gpuArray.rand(1,50,'single'));
F5 = 0.01*(gpuArray.rand(5,5,3,50,50,'single')); B5 = 0.01*(gpuArray.rand(1,50,'single'));
F7 = 0.01*(gpuArray.rand(3,3,3,50,50,'single')); B7 = 0.01*(gpuArray.rand(1,50,'single'));
F8 = 0.01*(gpuArray.rand(1,4,2,50,50,'single')); B8 = 0.01*(gpuArray.rand(1,50,'single'));
F9 = 0.01*(gpuArray.rand(1,1,1,50,K,'single')); B9 = 0.01*(gpuArray.rand(1,K,'single'));

%% fprop
res.x1 = mex_conv3d(X,F1,B1,'stride',c_str);                        % 1 Layer
[res.x2, ind2] = mex_maxpool3d(res.x1,'pool',pool,'stride',p_str);  % 2  Layer
res.x3 = mex_conv3d(res.x2,F3,B3,'stride',c_str);                   % 3  Layer
[res.x4, ind4] = mex_maxpool3d(res.x3,'pool',pool,'stride',p_str);  % 4  Layer
res.x5 = mex_conv3d(res.x4,F5,B5,'stride',c_str);                   % 5  Layer
[res.x6, ind6] = mex_maxpool3d(res.x5,'pool',pool,'stride',p_str);  % 6  Layer
res.x7 = mex_conv3d(res.x6,F7,B7,'stride',c_str);                   % 7  Layer
res.x8 = mex_conv3d(res.x7,F8,B8,'stride',c_str);                   % 8  Layer
res.x9 = mex_conv3d(res.x8,F9,B9,'stride',c_str);                   % 9  Layer
% y = vl_nnsoftmaxloss(res.x9,label); %10 Layer
tmpX = reshape(res.x9, K,N);
y = softmaxloss_fprop(tmpX, label);
%% bprop
% one = gpuArray(single(1)) ;
% res.dzdy = one;
% res.dzdx8 = vl_nnsoftmaxloss(res.x9,label,res.dzdy);%10

tmpdX = softmaxloss_bprop(tmpX, label);
res.dzdx9 = reshape(tmpdX, size(res.x9)); % need modification for code below?

[res.dzdx8,res.dzdF9,res.dzdB9] = mex_conv3d(res.x8,F9,B9, res.dzdx9,'stride',c_str);     % 9
[res.dzdx7,res.dzdF8,res.dzdB8] = mex_conv3d(res.x7,F8,B8, res.dzdx8,'stride',c_str);     % 8
[res.dzdx6,res.dzdF7,res.dzdB7] = mex_conv3d(res.x6,F7,B7, res.dzdx7,'stride',c_str);     % 7
res.dzdx5 = mex_maxpool3d(res.dzdx6,ind6,size(res.x5),'pool',pool,'stride',p_str);        % 6
[res.dzdx4,res.dzdF5,res.dzdB5] = mex_conv3d(res.x4,F5,B5, res.dzdx5,'stride',c_str);     % 5
res.dzdx3 = mex_maxpool3d(res.dzdx4,ind4,size(res.x3),'pool',pool,'stride',p_str);        % 4
[res.dzdx2,res.dzdF3,res.dzdB3] = mex_conv3d(res.x2,F3,B3, res.dzdx3,'stride',c_str);     % 3
res.dzdx1 = mex_maxpool3d(res.dzdx2,ind2,size(res.x1),'pool',pool,'stride',p_str);        % 2
[res.dzdx,res.dzdF1,res.dzdB1] = mex_conv3d(X,F1,B1, res.dzdx1,'stride',c_str);           % 1
