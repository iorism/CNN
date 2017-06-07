function sz = size5d(x)
%SIZE5D Summary of this function goes here
%   Detailed explanation goes here
  for i = 1 : 5
    sz(i) = size(x,i);
  end
  sz = sz(:)';
end

