function tc_3(dg)
desc = 'case 3, regular, pool > stride, non-zero pad';
%%
sz = [7,8,5, 5,9];
pool   = [3,2,4];
stride = [2,1,2];
pad    = [1,1,  0,0, 2,1];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();