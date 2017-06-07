%% data
ran = 50;
szX = [8,8,8, 5,9];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];

X = ran * randn(szX, 'single');
F = ran * randn(szF, 'single');
B = ran * randn(szB, 'single');

iX = [2,7,4, 3,5];
ep = 1e-2;

%% num appro
deltaX = zeros(szX, 'single');
deltaX(iX(1), iX(2), iX(3), iX(4), iX(5)) = ep .* ran;
%
Y2 = mex_conv3d( X + deltaX, F, B);
z2 = sum( Y2(:) );
%
Y1 = mex_conv3d( X - deltaX, F, B);
z1 = sum( Y1(:) );
%
dzdep_app = (z2 - z1)/(2*ep*ran);

%% by calculation
Y = mex_conv3d(X,F,B);
dzdy = ones(size(Y), 'single');
[dzdx,~,~] = mex_conv3d(X,F,B, dzdy);
dzdep = dzdx(iX(1), iX(2), iX(3), iX(4), iX(5));
%% compare
r = abs(dzdep_app - dzdep)/abs(dzdep);
fprintf('relative numeric diff: %4.3f %%\n', 100*r)