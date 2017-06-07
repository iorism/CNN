%MEX_MAXPOOL3D Volume max pooling for 3D convnet
%  [Y,ind] = MEX_MAXPOOL3D(X); forward pass
%  dZdX = MEX_MAXPOOL3D(dZdY, ind); backward pass (deprecated, will issue warning)
%  dzdX = MEX_MAXPOOL3D(dZdY, ind, szX); backward pass
%  [...] = MEX_MAXPOOL3D(..., 'pool',pool, 'stride',s, 'pad',pad); options
%
%  Input:
%   X: [H,W,D,M,N]. Volume at input port or feature maps. H, W, D are volume's 
%   height, width and depth, respectively. M is #volumes (or #feature maps).
%   N is #instances.
%   dZdY: [Ho,Wo,Do,M,N]. Delta signal at output port. Z means loss.
%   ind: see ind Output
%   szX: [5] or [4]. size of X or dzdX (they must be the same). append 
%   element 1 to its tail if numel(szX)==4
%
% Options:  
%   pool: [PH,PW,PD] or [P]. Pooling 3D window size. PH, PW, PD are the 
%   height, width and depth, respectively. P is the size for all. Default 
%   to P = 2.
%   s: [sH,sW,sD] or [s]. Default to s = 2
%   pad: [PH,PW,PD] or [P]. Padding size. Default to P = 0
%
%  Output:
%   Y: [Ho,Wo,Do,M,N]. Feature maps at output port
%   ind: [Ho,Wo,Do,M,N]. Linear index of the max elements to X so that 
%        Y = X(ind). ind must be int32 so that X cannot be too large, or
%        there would be undefined behaviour.
%   dZdX: [H,W,D,M,N]. Delta signal at input port. Z means loss.  
%