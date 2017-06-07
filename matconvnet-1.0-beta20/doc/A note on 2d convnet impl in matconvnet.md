The conventions and rationale of the convolution in matconvnet is explained in [the mannual](http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf). The im2row trick is used so that convolution is converted to matrix multiplication. We briefly explain it using Matlab syntax and the C/C++ code in matconvnet is no more than a 'literal translation'.

### FPROP
Given the input `x`, filter `f`, bias `B`:
``` 
x: [H,   W,   D,  N]
f: [H',  W',  D,  K]   B: [1, K]
```
Produce the output `y`:
```
y: [H'', W'', K,  N]
```

For each instance, convert them to matrix and multiply them as plain 2D matrix:
``` 
for i = 1 : N
  phix = im2row( x(:,:,:,i) );  % [H''W'', H'W'D] <-- [H, W, D]
  F    = reshape(f);            % [H'W'D, K] <-- [H', W', D, K] 
  Y(:,:,i) = phix * F;          % [H''W'', K] = [H''W'', H'W'D] * [H'W'D, K]
  
  u = ones([H''W'', 1]); 
  Y(:,:,i) += u * B;  % [H''W'', K] = [H''W'',1] * [1,K]
end
```
Thanks to the column-major element order in Matlab, `Y: [H''W'', K, N]` is already the desired `y: [H'', W'', K,  N] `. 

### BPROP
Given:
```
x: [H,   W,   D,  N]
f: [H',  W',  D,  K]
dy: [H'', W'', K,  N]
```
Produce:
```
df: [H', W', D, K], dB: [1, K]
dx: [H, W, D, N]
```

For each instance, accumulate on `dF`, `dB` and computes `dx`
``` 
for i = 1 : N
  phix = im2row( x(:,:,:,i) );    % [H''W'', H'W'D] <-- [H, W, D]
  dY   = reshape( dy(:,:,:,i) );  % [H''W'', K] <-- [H'', W'', K]
  dF += phix' * dY;               % [H'W'D, K] = [H'W'D, H''W''] * [H''W'', K]
  
  u = ones([H''W'', 1]);
  dB += u' * dY; % [1, K] = [1, H''W''] * [H''W'', K]
  
  dphix = dY * F' ;             % [H''W'', H'W'D] = [H''W'', K] * [K, H'W'D]
  dx(:,:,:,i) = row2im(dphix);  % [H, W, D] <-- [H''W'', H'W'D]
end
```

### im2row and row2im
For input feature map `x` and the corresponding filter bank `f` with stride 1 and pad 0, a 'stacked' matrix `phix =: im2row(x)` is generated:
```
x:    [H,   W,   D]
f:    [H',  W',  D]
y:    [H'', W'', 1]
phix: [H''W'', H'W'D]
```
where `H'' = H - H' + 1`, `W'' = W - W' + 1`, `y` is the expected filter respnose map that dosn't matter here but would be helpful as a placeholder/reminder. Specifically, the signature for `im2row`:
```
im2row(phix,
       x,
       H, W, D,
       H', W',
       filter-stride-on-H, filter-stride-on-W,
       pad-lower-on-H, pad-higher-on-H, pad-lower-on-W, pad-higher-onW)
```

Likewise, the 'inverse' conversion recovers matrix `x` from the stacked matrix `phix`: `x = row2im(phix)`. The signature:
```
row2im(x,
       phix,
       H, W, D,
       H', W',
       filter-stride-on-H, filter-stride-on-W,
       pad-lower-on-H, pad-higher-on-H, pad-lower-on-W, pad-higher-onW)
```

Remark:
- The argument names are different in `*.h` and `*.cpp`, don't get confused. In `im2row.hpp` the Matlab's column major convention is used, while in `im2row_xxx.cpp` the C/C++ row major convention is used.

### Matrix Multiplication
See the blas doc on [gemm](http://www.math.utah.edu/software/lapack/lapack-blas/sgemm.html) and [gemv](http://www.math.utah.edu/software/lapack/lapack-blas/sgemv.html). A brief explanation:
```
gemm(TA, TB,
     M, N, K,
     alpha,
     A, lda,
     B, ldb,
     beta,
     C, ldc) ;
```
It computes: `C = alpha*op(A)*op(B) + beta*C`, where
```
TA,TB: 't' transpose, op(A) = A' ; 'n' not transpose, op(A) = A; the same for B
op(A): [M, K]
op(B): [K, N]
C:     [M, N]
alpha, beta: [1] scalar
lda, ldb, ldc: stride of the leading dimension for A, B, C
```

```
gemv(TA,
    M, N,
    alpha,
    A, lda,
    x, incx,
    beta,
    y, incy)
```
It computes the transposed version `y = alpha * A' * x + beta * y` or non-transposed version `y = alpha * A * x + beta * y`, where
```
TA: `t` transpose (so that y's length is N); 'n' not transpose (so that y's length is M)
A: [M, N]
lda: stride on A's leading dimension
incx, incy: stride on the elements
alpha, beta: [1] scalar
```
Remark:
- `gemm` and `gemv` are inconsistent in terms of behaviours! Be careful with the transpose specification for `gemv` where the length of `y` is automatically decided! Also, in `gemv` the argument `A` is always `[M, N]`
- The matrix is *column major* which is consistent with Matlab conventions
- The stride `lda`, `ldb`, `ldc` account for sub matrix or memory that is not contiguous 
- The stride `incx` accounts for sub (row) vector
- In both `gemm` and `gemv`, `beta = 1` effectively accumulates the results; `beta = 0` overwrites the results (initialized with zeros)

### An example
Suppose a mini batch with the size:
```
x: [12, 12, 20, 100]
f: [5, 5, 20, 50];  B: [1, 50]
y: [8, 8, 50, 100]
```

For each instance:
```
x: [12, 12, 20]
f: [5,  5,  20, 50]; B: [1, 50]
y: [8,  8,  1,  50]
```

Do convolution-matrix conversion: `phiX = im2row(x)`. Do reshaping: `F = reshape(f)`, `Y = reshape(y)`.
So that 
```
phiX: [64, 500]
F:    [500, 50], B: [1, 50]
Y:    [64,  50]
```

#### FPROP
With the below size in mind:
```
X: [12, 12, 20]

phiX: [64,  500]
F:    [500, 50];  B: [1, 50]
Y:    [64,  50]
```
We have:

`Y = phiX * F + u * B`, where `u: [64, 1]` with all one elements.

The `gemm` call for `phiX * F`:
```
gemm('n', 'n',
     M = 64, N = 50, K = 500,
     alpha = 1,
     A = phiX, ldA = 64,
     B = F,    ldB = 500,
     beta = 0,
     C = Y, ldC = 64)
```

The `gemm` call for `Y += u*B`:
```
gemm('n', 'n'
     M = 64, N = 50, K = 1,
     alpha = 1,
     A = u, ldA = 64,
     B = B, ldB = 1,
     beta = 1,
     C = Y, ldC = 64)
```

#### BPROP
With the below size in mind:
```
X, dX: [12, 12, 20]

phiX, dphiX: [64,  500]
F, dF:       [500, 50];  B, dB: [1, 50]
Y, dY:       [64,  50]
```
We have:

`dF += phiX' * dY`. 
The corresponding `gemm` call:
```
gemm('t', 'n',
     M = 500, N = 50, K = 64,
     alpha = 1,
     A = phiX, ldA = 64,
     B = dY, ldB = 64,
     beta = 1 (or zero for the first instance with uninitialized dF),
     C = dF, ldC = 500)
```

`dB += u' * dY`, where `u: [1,64]` with all one elements. But to leverage `gemv`, it's convenient to compute the (equivalent) transposed version:
`dB' += dY' * u`. The corresponding `gemv` call:
```
gemv('t', 
     M = 500, N = 50,
     alpha = 1,
     A = dY, ldA = 64,
     x = u,  incx = 1,
     beta = 1 (or zero for the first instance with uninitialized dB),
     y = dB, incy = 1)
```

`dphiX = dY * F'`. 
The corresponding `gemm` call:
```
gemm('n','t',
     M = 64, N = 500, K = 64,
     alpha = 1,
     A = dY, ldA = 64,
     B = F, ldB = 500,
     beta = 0,
     C = dphiX, ldC = 64)
```

`dX = row2im(dphiX)`: `[12, 12, 20] <-- [64, 500]`

Remark:
- When computing `dF, dB` the derivative over each instance should be accumulated; When computing `dX` the derivative over each instance should be kept as is.
- In `gemm` or `gemv`, `beta = 0` means overwriting the memory so that one needs not initialize it with zeros; `beta = 1` effectively accumulates the results
