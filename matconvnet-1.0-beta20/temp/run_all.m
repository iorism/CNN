function run_all ()
%% conv 3d
dir_name = 't_c3d';
runInDir(dir_name);

%% max pooling 3d
dir_name = 't_mp3d';
runInDir(dir_name);

function runInDir (dir_name)

this_dir = fileparts( mfilename('fullpath') );
tar_dir  = fullfile(this_dir, ['+',dir_name]);

fns = dir(tar_dir);
for i = 1 : numel(fns)
  if ( fns(i).isdir ), continue; end
  
  % expect a test case with name 'tc_*'
  [~,nm,ext] = fileparts( fns(i).name );
  if ( strcmp(nm(1:3),'tc_') && strcmp(ext,'.m') )
    cmd = sprintf('%s.%s()', dir_name, nm);
    fprintf('running %s...\n', cmd);
    
    eval(cmd);
    
  end % if
  
end