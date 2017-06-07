function tc_6(dg)
desc = 'case 6, mod(sz+pad, pool*stride) > 0, abandon elements in X';
%%
sz     = [7, 12, 5, 50];
pool   = [2, 2, 2];
stride = [2, 2, 1];
pad    = [0,0, 0,0, 0,0];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();
%%
sz     = [5, 7, 3, 4, 12];
pool   = [2, 2, 2];
stride = [2, 2, 1];
pad    = [0,0, 0,0, 0,0];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();