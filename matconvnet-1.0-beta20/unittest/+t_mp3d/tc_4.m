function tc_4(dg)
desc = 'case 4, regular, pool > stride, non-zero pad';
%%
sz     = [9, 3, 5, 7,6];
pool   = [2, 3, 4];
stride = [2, 1, 2];
pad    = [1,0,  0,0, 3,0];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();