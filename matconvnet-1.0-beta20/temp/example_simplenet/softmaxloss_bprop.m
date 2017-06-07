function dzdX = softmaxloss_bprop(X, tar)
%softmaxloss_bprop Summary of this function goes here
%   Input:
%     X: [K, N] input data
%     tar: [K, N] target to be fitted, 0/1 response
%   Output:
%     dzdX: [K, N] the derivative, z stands for loss 
%   K: #classes, N: #instances
  
  assert( isequal(size(X), size(tar)) );
  
  % the probability estimation via the link function (softmax)
  expX = exp(X); % [K,N]
  s = sum(expX,1); % [1,N]
  p = bsxfun(@rdivide, expX, s); % [K,N]
  % the derivative
  dzdX = p - tar; % [K,N]
  
end

