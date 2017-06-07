%MEX_CONV3D Volume convolution for 3D convnet
% Y = MEX_CONV3D(X, F, B); forward pass
% [dZdX, dZdF, dZdB] = MEX_CONV3D(X, F, B dZdY); backward pass
% [...] =MEX_CONV3D(..., 'stride', s, 'pad', p); the options
%
% Input:
%   X: [H,W,D,P,N]. Input volume or feature maps. H, W, D are volume's 
%   height, width and depth, respectively. P is #volumes (or #feature maps).
%   N is #instances.
%   F: [FH,FW,FD,P,Q]. Filter kernels (a volume). FH, FW, FD are kernel's
%   height, width and depth, respectively. P is #input feature map and Q is
%   #output feature map
%   B: [1, Q]. Bias.
%   dZdY: [Ho,Wo,Do,Q,N]. Output feature maps. Subscript "o" means output.
%   s: [1] or [a,b,c]. the stride. default 1
%   p: [1] or [a,b,c]. pad. default 0
%
% Output:
%   Y: [Ho,Wo,Do,Q,N]. Output feature maps
%   dZdX: [H,W,D,P,N]. Delta w.r.t X where Z means loss. 
%   dZdF: [FH,FW,FD,P,Q]. Delta w.r.t. F
%   dZdB: [1, Q]. Delta w.r.t. B
%