function A = getSeqMat( sz )
%GETSEQMAT Summary of this function goes here
%   Detailed explanation goes here
  A = 1 : prod(sz);
  A = reshape(A, sz);
  A = single(A);
end

