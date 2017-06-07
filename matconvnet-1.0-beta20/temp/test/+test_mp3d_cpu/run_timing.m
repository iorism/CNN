function run_timing(  )
sz = [32,32,32, 5, 16];

T = 6;
fprintf('timing for %s : %s ',...
  test_mp3d_cpu.get_name(),...
  get_szStr(sz) );
te = tic;
for t = 1 : T
  test_mp3d_cpu.testcase( sz );
end
te = toc(te);
fprintf(' avg time = %6.2f\n', te/T);
