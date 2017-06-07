function testcase(sz)

x = rand(sz, 'single');
[y, ind] = mex_maxpool3d(x);

xx = mex_maxpool3d(y,ind);


end

