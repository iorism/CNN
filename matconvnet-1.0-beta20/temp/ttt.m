%%
H = reshape( (1:9), 3,3);
FF = convmtx2(H, [5,5]);
%%
A = reshape( (1:24), 4,6);
B = im2col(A, [2,2]);