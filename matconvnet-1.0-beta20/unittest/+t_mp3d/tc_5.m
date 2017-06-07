function tc_5(dg)
desc = 'case 5, regular, pool = 2*stride, non-zero pad';
%%
sz     = [30, 30, 30, 5, 16];
pool   = [4, 4, 4];
stride = [2, 2, 2];
pad    = [1,1,  1,1, 1,1];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();