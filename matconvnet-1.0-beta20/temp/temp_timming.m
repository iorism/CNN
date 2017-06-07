%%
a = gpuArray( rand(3,4,5,6,7, 'single') );
b = rand(7,6,5,4,3, 'single');
%%
fprintf('gpu array ');
T = 1e6;
te = tic;
for t = 1 : T
  temp(a);
end
te = toc(te);
fprintf('avg time = %d\n', te/T);
%%
fprintf('cpu array ');
T = 1e6;
te = tic;
for t = 1 : T
  temp(b);
end
te = toc(te);
fprintf('avg time = %d\n', te/T);
