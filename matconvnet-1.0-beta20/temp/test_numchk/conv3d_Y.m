%% data
szX = [8,8,8, 5,9];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];
X = rand(szX, 'single');
F = rand(szF, 'single');
B = rand(szB, 'single');
%% fprop
Y = mex_conv3d(X,F,B);
%% bprop
dZdY = rand(size(Y), 'single');
[dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY);
%% validate size
% size Y
szY123 = szX(1:3)- szF(1:3) + 1;
szY = [szY123, szF(5), szX(5)];
assert( all( szY == size(Y) ) );

% size dX, dF, dB
assert( all( size(X)==size(dZdX) ) );
assert( all( size(F)==size(dZdF) ) );
assert( all( size(B)==size(dZdB) ) );
%% validate results: Y, fprop
tol = 1e-8;
% instance count
iInst = 4;
% output feature map count
q = 3; 
% offset on X, index on Y
% beg = [6,7,7]; 
% beg = [1,1,1]; 
beg = [3,3,5]; 

%
xx = X(...
  beg(1) : beg(1) + szF(1) - 1,...
  beg(2) : beg(2) + szF(2) - 1,...
  beg(3) : beg(3) + szF(3) - 1,...
  :,...
  iInst);
ff = F(:,:,:,:, q);
bb = B(1, q);
% calculated result
yy = Y(beg(1), beg(2), beg(3), q, iInst);     
% expected result
yy2 = sum( xx(:).*ff(:) ) + bb;
%
% assert( abs(yy-yy2) < eps );
fprintf('numeric diff: %10.9f\n',...
  abs(yy-yy2));