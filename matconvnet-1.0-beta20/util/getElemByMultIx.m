function r = getElemByMultIx(X, ix)
  
  ix = ix(:);
  r = [];
 
  for i = 1 : numel(ix)
    if( ix(i) == 0), continue; end
    
    for j =  1 : ix(i)
      r(end+1) = X(i);
    end
  end

end