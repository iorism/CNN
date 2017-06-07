function tc_2(dg)
desc = 'case 2, regular, pool = stride, non-zero pad';
%%
sz = [7,8,5, 5,9];
pool   = [3,2,4];
stride = [3,2,4];
pad    = [1,1,  0,0, 2,1];

h = dg(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();