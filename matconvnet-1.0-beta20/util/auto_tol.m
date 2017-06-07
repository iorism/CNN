function tau = auto_tol(a, b)
%AUTO_TOL Summary of this function goes here
%   Adapted from VL_TESSIM in vlfeat/matconvnet
% VL_TESSIM  Test near-equality of arrays
%   VL_TEST(A,B,TAU) succeds if A and B have the same dimensions
%   and if their L^infinity difference is smaller than TAU.
%
%   VL_TEST(A,B) selects TAU automatically by looking at the
%   dynamic range of the data. The same happens if TAU is the empty
%   matrix.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


maxv = max([max(a(:)), max(b(:))]) ;
minv = min([min(a(:)), min(b(:))]) ;
tau = 1e-2 * (maxv - minv) + 1e-4 * max(maxv, -minv) ;

% tau = 1e-1 * (maxv - minv) + 1e-3 * max(maxv, -minv) ;


