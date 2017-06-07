function tc_1(dg)
desc = 'case 1, regular, pool = stride, no pad';
sz = [8,8,8, 5,9];
pool   = [2, 2, 2];
stride = [2, 2, 2];
pad = [0 0 0 0 0 0];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();
%%
sz = [4,2,2, 1,1];
pool   = [2, 2, 2];
stride = [2, 2, 2];
pad = [0 0 0 0 0 0];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();