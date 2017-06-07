function tc_1(dg)
desc = 'case 1, regular, no padding, stride 1';
%%
% szX = [2, 2, 1, 1, 1];
% szF = [2, 2, 1, 1, 1];
% szB = [1, 1];
% stride = [1,1,1];
% pad = [0,0, 0,0, 0,0];
% 
% h = dg(szX,szF,szB, stride, pad, desc);
% 
% hu = t_c3d.unit(h);
% hu.run();
%%
szX = [9, 9, 1, 1, 1];
szF = [3, 3, 1, 1, 1];
szB = [1, 1];
stride = [1,1,1];
pad = [0,0, 0,0, 0,0];

h = dg(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();
%%
szX = [7, 5, 6, 2, 1];
szF = [2, 3, 4, 2, 3];
szB = [1, 3];
stride = [1,1,1];
pad = [0,0, 0,0, 0,0];

h = dg(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();
%%
szX = [4, 3, 5, 4, 1];
szF = [2, 1, 3, 4, 5];
szB = [1, 5];
stride = [1,1,1];
pad = [0,0, 0,0, 0,0];

h = dg(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();
%%
szX = [16, 8, 4, 5, 3];
szF = [2, 4, 3, 5, 2];
szB = [1, 2];
stride = [1,1,1];
pad = [0,0, 0,0, 0,0];

h = dg(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();