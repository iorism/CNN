function Y = softmaxloss_fprop(X, tar)
%softmaxloss_fprop Summary of this function goes here
%   Input:
%     X: [K, N] input data
%     tar: [K, N] target to be fitted, 0/1 response
%   Output:
%     Y: [1, N] the loss 
%   K: #classes, N: #instances
  
  assert( isequal(size(X), size(tar)) );
  
  % the probability estimation via the link function (softmax)
  expX = exp(X); % [K,N]
  s = eps + sum(expX,1); % [1,N]
  p = bsxfun(@rdivide, expX, s); % [K,N]
  % the loss
  Y = sum( -tar .* log(eps+p) ); % [1,N]
  
  % note: don't do the average by N
end

