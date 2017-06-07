As is the case of 2d convnet, the `im2row` (and `row2im`) trick is used to convert convolution to plain matrix multiplication. 5 dimensional array shows up: for input `x` or output `y`, dim1 to dim3 are height, width, depth; dim4 is #input/#output channels; dim5 is #instances. For filter bank `f`, dim4 is #input-feature-maps, dim4 is #output-feature-maps.

## General Case
### FPROP
Given the input `x`, filter `f`, bias `B`:
``` 
x: [H,   W,   D,  P, N]
f: [H',  W',  D', P, Q]   B: [1, Q]
```
Produce the output `y`:
```
y: [H'', W'', D'', Q, N]
```

For each instance, convert the data to matrix and do a plain 2D matrix multiplication:
``` 
for i = 1 : N
  phix = im2row( x(:,:,:,:, i) );  % [H''W''D'', H'W'D'P] <-- [H, W, D, P]
  F    = reshape(f);               % [H'W'D'P, Q] <-- [H', W', D', P, Q] 
  Y(:,:, i) = phix * F;            % [H''W''D'', Q] = [H''W''D'', H'W'D'P] * [H'W'D'P, Q]
  
  u = ones([H''W''D'', 1]); 
  Y(:,:,i) += u * B;  % [H''W''D'', Q] = [H''W''D'', 1] * [1, Q]
end
```
Thanks to the column-major element order in Matlab, the 3D matrix `Y: [H''W''D'', Q, N]` is already the desired 5D matrix `y: [H'', W'', D'', Q,  N] `. 

### BPROP
For notation simplicity, we ommit `dz` in any term `dzdx` where `z` means the scalar loss. For example, `dzdx` is abbreviated as `dx`.

Given:
```
x:  [H,   W,   D,   P,  N]
f:  [H',  W',  D',  P,  Q], B [1, Q]
dy: [H'', W'', D'', Q,  N]
```
Produce:
```
df: [H', W', D', P, Q],  dB: [1, Q]
dx: [H,  W,  D,  P, N]
```

For each instance, **accumulate** on `dF`, `dB` and computes `dx`
``` 
for i = 1 : N
  phix = im2row( x(:,:,:,:, i) );     % [H''W''D'', H'W'D'P] <-- [H, W, D, P]
  dY   = reshape( dy(:,:,:,:, i) );   % [H''W''D'', Q] <-- [H'', W'', D'', Q]
  dF += phix' * dY;                   % [H'W'D'P, Q] = [H'W'D'P, H''W''D''] * [H''W''D'', Q]
  
  u = ones([H''W''D'', 1]);
  dB += u' * dY;  % [1, Q] = [1, H''W''D''] * [H''W''D'', Q]
  
  dphix = dY * F' ;             % [H''W''D'', H'W'D'P] = [H''W''D'', Q] * [Q, H'W'D'P]
  dx(:,:,:,:, i) = row2im(dphix);  % [H, W, D, P] <-- [H''W''D'', H'W'D'P]
end
```

## A (Concrete) Example
### FPROP
Input:
``` 
x: [8, 8, 8, 5, 9]
f: [3, 2, 2, 5, 4]   B: [1, 4]
```
Produce:
```
y: [6, 7, 7, 4, 9]
```


``` 
6*7*7 = 294
3*2*2*5 = 60

for i = 1 : 9
  phix = im2row( x(:,:,:,:, i) );  % [294, 60] <-- [8, 8, 8, 5]
  F    = reshape(f);               % [60, 4] <-- [3, 2, 2, 5, 4] 
  Y(:,:, i) = phix * F;            % [294, 4] = [294, 60] * [60, 4]
  
  u = ones([294, 1]); 
  Y(:,:,i) += u * B;  % [294, 4] = [294, 1] * [1, 4]
end
```
Thanks to the column-major element order in Matlab, the 3D matrix `Y: [294, 4, 9]` is already the desired 5D matrix `y: [6, 7, 7, 4, 9] `. 

### BPROP
Given:
```
x:  [8, 8, 8, 5, 9]
f:  [3, 2, 2, 5, 4], B: [1, 4]
dy: [6, 7, 7, 4, 9]
```
Produce:
```
df: [3, 2, 2, 5, 4],  dB: [1, 4]
dx: [8, 8, 8, 5, 9]
```

For each instance, **accumulate** on `dF`, `dB` and computes `dx`
``` 
6*7*7 = 294
3*2*2*5 = 60

for i = 1 : 9
  phix = im2row( x(:,:,:,:, i) );     % [294, 60] <-- [8, 8, 8, 5]
  dY   = reshape( dy(:,:,:,:, i) );   % [294, 4] <-- [6, 7, 7, 4]
  dF += phix' * dY;                   % [60, 4] = [60, 294] * [294, 4]
  
  u = ones([294, 1]);
  dB += u' * dY;  % [1, 4] = [1, 294] * [294, 4]
  
  dphix = dY * F' ;               % [294, 60] = [294, 4] * [4, 60]
  dx(:,:,:,:, i) = row2im(dphix); % [8, 8, 8, 5] <-- [294, 60]
end
```

## Full Connection
When `X` and `F` matches in size and the padding is zero, convolution boils down to plain matrix multiplication and the loop over instances can be eliminated. The size is like
```
X: [H, W, D, P, N]
F: [H, W, D, P, Q]  B: [1, Q]
Y: [1, 1, 1, Q, N]
```

### FPROP
Given:
```
XX: [HWDP, N]
FF: [HWDP, Q]
BB: [Q, 1]
uu: [1, N] the all one vector
``` 
Produce:
```
YY: [Q, N]
```
by
```
YY =  FF' * XX
YY += BB * uu 
```

### BPROP
Given:
```
dYY:[Q, N]
XX: [HWDP, N]
FF: [HWDP, Q]
uu: [1, N] the all one vector
```
Produce:
```
dXX = FF * dYY
dFF = XX * dYY'
dBB = uu * dYY'
```