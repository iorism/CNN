
%% data
ran = 100;
szX = [8,8,8, 5,9];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];

X = ran * randn(szX, 'single');
F = ran * randn(szF, 'single');
B = ran * randn(szB, 'single');

iB = 2;
ep = 1e-1;

%% num appro
deltaB = zeros(szB, 'single');
deltaB(iB) = ep .* ran;
%
Y2 = mex_conv3d( X, F, B + deltaB);
z2 = sum( Y2(:) );
%
Y1 = mex_conv3d( X, F, B - deltaB);
z1 = sum( Y1(:) );
%
dzdep_app = (z2 - z1)/(2*ep*ran);

%% by calculation
Y = mex_conv3d(X,F,B);
dzdy = ones(size(Y), 'single');
[~,~,dzdB] = mex_conv3d(X,F,B, dzdy);
dzdep = dzdB(iB);
%% compare
r = abs(dzdep_app - dzdep)/abs(dzdep);
fprintf('relative numeric diff: %4.3f %%\n', 100*r)