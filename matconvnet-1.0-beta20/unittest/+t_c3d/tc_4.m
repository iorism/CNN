function tc_4(dg)
desc = 'case 4, regular, non-zero pad, stride bigger than 1';
%%
szX = [8, 8, 8, 5, 2];
szF = [3, 2, 2, 5, 4];
szB = [1, 4];
stride = [2,3,4];
pad = [5,1, 2,0, 3,4];

h = dg(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();
