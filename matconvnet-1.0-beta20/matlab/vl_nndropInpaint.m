function [y,mask] = vl_nndropInpaint(x,varargin)

opts.rate = 0.6 ;
opts.mask = [] ;

backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  opts = vl_argparse(opts, varargin) ;
end

% determine mask
mask = opts.mask ;
% scale = 1;
if backMode && isempty(mask)
  warning('vl_nndropInpaint: when using in backward mode, the mask should be specified') ;
end
if isempty(mask)
%   if isa(x,'gpuArray')
%     mask = scale * (gpuArray.rand(size(x), classUnderlying(x)) >= opts.rate) ;
%   else
%     mask = scale * (rand(size(x), 'like', x) >= opts.rate) ;
%   end
%     mask = gather(x(:,:,1,:));
    mask = x(:,:,1,:);
    idxZ = find(mask<opts.rate);
    idxR = find(mask>=opts.rate);
    mask(idxZ) = 0;
    mask(idxR) = 1;    
end

% do job
if ~backMode  
    y(:,:,1,:) = mask .* x(:,:,1,:);
%   y(:,:,1,:) = x(:,:,1,:);  %mask
%   y(:,:,2,:) = mask .* x(:,:,2,:); %rval
else
    y(:,:,1,:) = mask .* dzdy(:,:,1,:); %rval    
%   y(:,:,1,:) = dzdy(:,:,1,:);  %mask
%   y(:,:,2,:) = mask .* dzdy(:,:,2,:); %rval    
end
