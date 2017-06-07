%% data
ran = 100;
szX = [8,8,8, 5,9];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];

X = ran * randn(szX, 'single');
F = ran * randn(szF, 'single');
B = ran * randn(szB, 'single');

iF = [2,1,2, 3,2];
ep = 1e-2;

%% num appro
deltaF = zeros(szF, 'single');
deltaF(iF(1), iF(2), iF(3), iF(4), iF(5)) = ep .* ran;
%
Y2 = mex_conv3d( X, F + deltaF, B);
z2 = sum( Y2(:) );
%
Y1 = mex_conv3d( X, F - deltaF, B);
z1 = sum( Y1(:) );
%
dzdep_app = (z2 - z1)/(2*ep*ran);

%% by calculation
Y = mex_conv3d(X,F,B);
dzdy = ones(size(Y), 'single');
[~,dzdF,~] = mex_conv3d(X,F,B, dzdy);
dzdep = dzdF(iF(1), iF(2), iF(3), iF(4), iF(5));
%% compare
r = abs(dzdep_app - dzdep)/abs(dzdep);
fprintf('relative numeric diff: %4.3f %%\n', 100*r)