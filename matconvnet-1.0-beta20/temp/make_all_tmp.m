function make_all_tmp()
% adopted from the VL_COMPILENN.m which is part of MatConvNet toolbox

% Copyright (C) 2014 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Get this root directory
root = fileparts(mfilename('fullpath')) ;

% --------------------------------------------------------------------
%                                                        Parse options
% --------------------------------------------------------------------

opts.enableGpu        = true;
opts.enableCudnn      = false;
opts.enableOpenmp     = true;
opts.verbose          = 1;
opts.debug            = true;
opts.printLog         = true;
opts.cudaRoot         = [] ;
opts.cudaArch         = [] ;
opts.defCudaArch      = [...
  '-gencode=arch=compute_20,code=\"sm_20,compute_20\" '...
  '-gencode=arch=compute_30,code=\"sm_30,compute_30\"'];
opts.cudnnRoot        = 'local' ;

if opts.enableGpu
  opts.cudaMethod = 'nvcc' ;
else
  opts.cudaMethod = '';
end

% --------------------------------------------------------------------
%                                                     Files to compile
% --------------------------------------------------------------------

%%% mex gateway source
mex_src = {} ; 
mex_src{end+1} = fullfile('temp2.cu');


%%% lib source
lib_src = {} ; 
lib_src{end+1} = fullfile('../src/wrapperMx.cpp') ;



% --------------------------------------------------------------------
%                                                   Setup CUDA toolkit
% --------------------------------------------------------------------
arch = computer('arch') ;

if opts.enableGpu
  if isempty(opts.cudaRoot), opts.cudaRoot = search_cuda_devkit(opts); end
  check_nvcc(opts.cudaRoot);
  if opts.verbose 
    fprintf('%s:\tCUDA: using CUDA Devkit ''%s''.\n', ...
                          mfilename, opts.cudaRoot) ;
  end
  switch arch
    case 'win64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib', 'x64') ;
    case 'maci64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib') ;
    case 'glnxa64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib64') ;
    otherwise, error('Unsupported architecture ''%s''.', arch) ;
  end
  opts.nvccPath = fullfile(opts.cudaRoot, 'bin', 'nvcc') ;

  % CUDA arch string (select GPU architecture)
  if isempty(opts.cudaArch), opts.cudaArch = get_cuda_arch(opts) ; end
    if opts.verbose 
      fprintf('%s:\tCUDA: NVCC architecture string: ''%s''.\n', ...
         mfilename, opts.cudaArch) ;
    end
  % Make sure NVCC is visible by MEX by setting the corresp. env. var
  if ~strcmp(getenv('MW_NVCC_PATH'), opts.nvccPath)
    warning('Setting the ''MW_NVCC_PATH'' environment variable to ''%s''', ...
            opts.nvccPath) ;
    setenv('MW_NVCC_PATH', opts.nvccPath) ;
  end
end

% --------------------------------------------------------------------
%                                                     Compiler options
% --------------------------------------------------------------------

% Build directories
mex_dir = '.' ;
bld_dir = '.build' ;

% Common compiler flags: cc
flags.cc = {} ;
if opts.printLog
  flags.cc{end+1} = '-DVB';
end
if opts.verbose > 1, flags.cc{end+1} = '-v' ; end
if opts.enableGpu
  flags.cc{end+1} = '-DWITH_GPUARRAY' ; 
end
if opts.enableCudnn
  flags.cc{end+1} = '-DENABLE_CUDNN' ;
  flags.cc{end+1} = ['-I' opts.cudnnRoot] ;
end
flags.cc{end+1} = '-I"./src"';
if opts.debug
  if ( ~strcmp(opts.cudaMethod,'nvcc') )
    flags.cc{end+1} = '-g' ;
  end
else
  flags.cc{end+1} = '-DNDEBUG' ;
end

% Linker flags
flags.link = {} ;
flags.link{end+1} = '-largeArrayDims';
flags.link{end+1} = '-lmwblas' ;
if opts.debug
  flags.link{end+1} = '-g';
end
if opts.enableGpu
  flags.link{end+1} = ['-L' opts.cudaLibDir] ;
  flags.link{end+1} = '-lcudart' ;
  flags.link{end+1} = '-lcublas' ;
  switch arch
    case {'maci64', 'glnxa64'}
      flags.link{end+1} = '-lmwgpu' ;
    case 'win64'
      flags.link{end+1} = '-lgpu' ;
  end
  if opts.enableCudnn
    flags.link{end+1} = ['-L' opts.cudnnRoot] ;
    flags.link{end+1} = '-lcudnn' ;
  end
end

% For the MEX command: mexcc
flags.mexcc = flags.cc ;
flags.mexcc{end+1} = '-cxx' ;
if strcmp(arch, 'maci64')
  flags.mexcc{end+1} = 'CXXFLAGS=$CXXFLAGS -stdlib=libstdc++' ;
  flags.link{end+1} = 'LDFLAGS=$LDFLAGS -stdlib=libstdc++' ;
end
if opts.enableGpu 
  flags.mexcc{end+1} = ...
    ['-I"',...
    fullfile(matlabroot, 'toolbox','distcomp','gpu','extern','include'),...
    '"'] ;
end

% For the MEX command: mexcu
if opts.enableGpu 
  flags.mexcu = flags.cc ;
  flags.mexcu{end+1} = '-cxx' ;
  flags.mexcu(end+1:end+2) = {'-f' mex_cuda_config(root)} ;
  flags.mexcu{end+1} = ['NVCCFLAGS=' opts.cudaArch '$NVCC_FLAGS'] ;
end

% For the cudaMethod='nvcc': nvcc
if opts.enableGpu && strcmp(opts.cudaMethod,'nvcc')
  flags.nvcc = flags.cc;
  if opts.debug
    flags.nvcc{end+1} = '-g -G';
    flags.nvcc{end+1} = '-O0' ;
  end
  flags.nvcc{end+1} = ['-I"' fullfile(matlabroot, 'extern', 'include') '"'] ;
  flags.nvcc{end+1} = ['-I"' fullfile(matlabroot, 'toolbox','distcomp','gpu','extern','include') '"'] ;
  flags.nvcc{end+1} = '-Xcompiler' ;

  switch arch
    case {'maci64', 'glnxa64'}
      flags.nvcc{end+1} = '-fPIC' ;
    case 'win64'
      flags.nvcc{end+1} = '/MD' ;
      check_clpath(); % check whether cl.exe in path
  end
end

% For -largeArrayDims
flags.mexcc{end+1} = '-largeArrayDims';
if opts.enableGpu 
  flags.mexcu{end+1} = '-largeArrayDims';
end

% For openmp: mexcc
if opts.enableOpenmp
  if ( strcmp(arch(1:3), 'win') )
    flags.mexcc{end+1} = 'COMPFLAGS=/openmp $COMPFLAGS';
  else
    flags.mexcc{end+1} = 'CXXFLAGS="\$CXXFLAGS -fopenmp"';
  end
end

% For -largeArrayDims
flags.mexcc{end+1} = '-largeArrayDims';

% Verbose printing
if opts.verbose
  fprintf('%s: intermediate build products directory: %s\n', mfilename, bld_dir) ;
  fprintf('%s: MEX files: %s/\n', mfilename, mex_dir) ;
  fprintf('%s: MEX compiler options: %s\n', mfilename, strjoin(flags.mexcc)) ;
  fprintf('%s: MEX linker options: %s\n', mfilename, strjoin(flags.link)) ;
end
if ( opts.verbose && opts.enableGpu )
  fprintf('%s: MEX compiler options (CUDA): %s\n', ...
    mfilename, strjoin(flags.mexcu)) ;
end
if ( opts.verbose && opts.enableGpu && strcmp(opts.cudaMethod,'nvcc') )
  fprintf('%s: NVCC compiler options: %s\n', ...
    mfilename, strjoin(flags.nvcc)) ;
end

% --------------------------------------------------------------------
%                                                              Compile
% --------------------------------------------------------------------

% convert all to absolute path
mex_src = cellfun( @(x) fullfile(root,x),...
  mex_src, 'UniformOutput',false);
lib_src = cellfun( @(x) fullfile(root,x),...
  lib_src, 'UniformOutput',false);
mex_dir = fullfile(root, mex_dir) ;
bld_dir = fullfile(root, bld_dir) ;
if (~exist(bld_dir,'dir')), mkdir(bld_dir); end

% Intermediate object files
srcs = horzcat(lib_src,mex_src) ;
for i = 1 : numel( srcs )
  if strcmp(opts.cudaMethod,'nvcc')
    nvcc_compile(opts, srcs{i}, toobj(bld_dir,srcs{i}), flags.nvcc) ;
  else
    mex_compile(opts, srcs{i}, toobj(bld_dir,srcs{i}), flags.mexcc) ;
  end
end

% Link into MEX files
for i = 1:numel(mex_src)
  objs{i} = toobj(bld_dir, [mex_src{i}, lib_src]) ;
  mex_link(opts, objs{i}, mex_dir, flags.link) ;
end

% delete interdeidate files and directory
if ( ~opts.debug )
  cellfun( @(x) delobj(x), objs);
  rmdir( bld_dir );
end

% --------------------------------------------------------------------
%                                                    Utility functions
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function objs = toobj(bld_dir,srcs)
% --------------------------------------------------------------------
root = fileparts(mfilename('fullpath')) ;
objs = strrep(srcs,fullfile(root,'src'),bld_dir) ;
objs = strrep(objs,'.cpp',['.' objext()]) ;
objs = strrep(objs,'.cu',['.' objext()]) ;
objs = strrep(objs,'.c',['.' objext()]) ;

% --------------------------------------------------------------------
function delobj(objfiles)
% --------------------------------------------------------------------
% suppress the warning
for i = 1 : numel(objfiles)
  if (~exist(objfiles{i},'file')), continue; end
  delete( objfiles{i} );
end
%cellfun( @(x) delete(x), objfiles);

% --------------------------------------------------------------------
function mex_compile(opts, src, tgt, mex_opts)
% --------------------------------------------------------------------
mopts = {'-outdir', fileparts(tgt), src, '-c', mex_opts{:}} ;
if opts.verbose 
  fprintf('%s: MEX: %s\n', mfilename, strjoin(mopts)) ;
end
mex(mopts{:}) ;

% --------------------------------------------------------------------
function nvcc_compile(opts, src, tgt, nvcc_opts)
% --------------------------------------------------------------------
nvcc_path = fullfile(opts.cudaRoot, 'bin', 'nvcc');
nvcc_cmd = sprintf('"%s" -c "%s" %s -o "%s"', ...
                   nvcc_path, src, ...
                   strjoin(nvcc_opts), tgt);
if opts.verbose 
  fprintf('%s: CUDA: %s\n', mfilename, nvcc_cmd) ;
end
status = system(nvcc_cmd);
if status, error('Command %s failed.', nvcc_cmd); end;

% --------------------------------------------------------------------
function mex_link(opts, objs, mex_dir, mex_flags)
% --------------------------------------------------------------------
mopts = {'-outdir', mex_dir, mex_flags{:}, objs{:}} ;
if opts.verbose 
  fprintf('%s: MEX linking: %s\n', mfilename, strjoin(mopts)) ;
end
mex(mopts{:}) ;

% --------------------------------------------------------------------
function ext = objext()
% --------------------------------------------------------------------
% Get the extension for an 'object' file for the current computer
% architecture
switch computer('arch')
  case 'win64', ext = 'obj';
  case {'maci64', 'glnxa64'}, ext = 'o' ;
  otherwise, error('Unsupported architecture %s.', computer) ;
end

% --------------------------------------------------------------------
function conf_file = mex_cuda_config(root)
% --------------------------------------------------------------------
% Get mex CUDA config file
mver = [1e4 1e2 1] * sscanf(version, '%d.%d.%d') ;
if mver <= 80200, ext = 'sh' ; else ext = 'xml' ; end
arch = computer('arch') ;
switch arch
  case {'win64'}
    config_dir = fullfile(matlabroot, 'toolbox', ...
                          'distcomp', 'gpu', 'extern', ...
                          'src', 'mex', arch) ;
  case {'maci64', 'glnxa64'}
    config_dir = fullfile(root, 'matlab', 'src', 'config') ;
end
conf_file = fullfile(config_dir, ['mex_CUDA_' arch '.' ext]);
fprintf('%s:\tCUDA: MEX config file: ''%s''\n', mfilename, conf_file);

% --------------------------------------------------------------------
function check_clpath()
% --------------------------------------------------------------------
% Checks whether the cl.exe is in the path (needed for the nvcc). If
% not, tries to guess the location out of mex configuration.
[status,~] = system('cl.exe -help');
if (status ~= 0)
  warning('CL.EXE not found in PATH. Trying to guess out of mex setup.');
  cc = mex.getCompilerConfigurations('c++');
  if isempty(cc)
    error('Mex is not configured. Run "mex -setup".');
  end
  prev_path = getenv('PATH');
  cl_path = fullfile(cc.Location, 'VC','bin', cc.Details.CommandLineShellArg);
  setenv('PATH', [prev_path ';' cl_path]);
  [status,~] = system('cl.exe');
  if (status ~= 0)
    setenv('PATH', prev_path);
    error('Unable to find cl.exe');
  else
    fprintf('Location of cl.exe (%s) successfully added to your PATH.\n', ...
      cl_path);
  end
end

% -------------------------------------------------------------------------
function paths = which_nvcc(opts)
% -------------------------------------------------------------------------
switch computer('arch')
  case 'win64'
    [~, paths] = system('where nvcc.exe');
    paths = strtrim(paths);
    paths = paths(strfind(paths, '.exe'));
  case {'maci64', 'glnxa64'}
    [~, paths] = system('which nvcc');
    paths = strtrim(paths) ;
end

% -------------------------------------------------------------------------
function cuda_root = search_cuda_devkit(opts)
% -------------------------------------------------------------------------
% This function tries to to locate a working copy of the CUDA Devkit.

opts.verbose && fprintf(['%s:\tCUDA: seraching for the CUDA Devkit' ...
                    ' (use the option ''CudaRoot'' to override):\n'], mfilename);

% Propose a number of candidate paths for NVCC
paths = {getenv('MW_NVCC_PATH')} ;
paths = [paths, which_nvcc(opts)] ;
for v = {'5.5', '6.0', '6.5', '7.0'}
  switch computer('arch')
    case 'glnxa64'
      paths{end+1} = sprintf('/usr/local/cuda-%s/bin/nvcc', char(v)) ;
    case 'maci64'
      paths{end+1} = sprintf('/Developer/NVIDIA/CUDA-%s/bin/nvcc', char(v)) ;
    case 'win64'
      paths{end+1} = sprintf('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%s\\bin\\nvcc.exe', char(v)) ;
  end
end
paths{end+1} = sprintf('/usr/local/cuda/bin/nvcc') ;

% Validate each candidate NVCC path
for i=1:numel(paths)
  nvcc(i).path = paths{i} ;
  [nvcc(i).isvalid, nvcc(i).version] = validate_nvcc(opts,paths{i}) ;
end
if opts.verbose
  fprintf('\t| %5s | %5s | %-70s |\n', 'valid', 'ver', 'NVCC path') ;
  for i=1:numel(paths)
    fprintf('\t| %5d | %5d | %-70s |\n', ...
            nvcc(i).isvalid, nvcc(i).version, nvcc(i).path) ;
  end
end

% Pick an entry
index = find([nvcc.isvalid]) ;
if isempty(index)
  error('Could not find a valid NVCC executable\n') ;
end
nvcc = nvcc(index(1)) ;
cuda_root = fileparts(fileparts(nvcc.path)) ;

if opts.verbose
  fprintf('%s:\tCUDA: choosing NVCC compiler ''%s'' (version %d)\n', ...
          mfilename, nvcc.path, nvcc.version) ;
end

% -------------------------------------------------------------------------
function [valid, cuver]  = validate_nvcc(opts, nvcc_path)
% -------------------------------------------------------------------------
valid = false ;
cuver = 0 ;
if ~isempty(nvcc_path)
  [status, output] = system(sprintf('"%s" --version', nvcc_path)) ;
  valid = (status == 0) ;
end
if ~valid, return ; end
match = regexp(output, 'V(\d+\.\d+\.\d+)', 'match') ;
if isempty(match), valid = false ; return ; end
cuver = [1e4 1e2 1] * sscanf(match{1}, 'V%d.%d.%d') ;

% --------------------------------------------------------------------
function check_nvcc(cuda_root)
% --------------------------------------------------------------------
% Checks whether the nvcc is in the path. If not, guessed out of CudaRoot.
[status, ~] = system('nvcc --help');
if status ~= 0
  warning('nvcc not found in PATH. Trying to guess out of CudaRoot.');
  cuda_bin_path = fullfile(cuda_root, 'bin');
  prev_path = getenv('PATH');
  switch computer
    case 'PCWIN64', separator = ';';
    case {'GLNXA64', 'MACI64'}, separator = ':';
  end
  setenv('PATH', [prev_path separator cuda_bin_path]);
  [status, ~] = system('nvcc --help');
  if status ~= 0
    setenv('PATH', prev_path);
    error('Unable to find nvcc.');
  else
    fprintf('Location of nvcc (%s) added to your PATH.\n', cuda_bin_path);
  end
end

% --------------------------------------------------------------------
function cudaArch = get_cuda_arch(opts)
% --------------------------------------------------------------------
if opts.verbose 
  fprintf(['%s:\tCUDA: ',...
    'determining GPU compute capability ',...
    '(use the ''CudaArch'' option to override)\n'], mfilename);
end
try
  gpu_device = gpuDevice();
  arch_code = strrep(gpu_device.ComputeCapability, '.', '');
  cudaArch = ...
      sprintf('-gencode=arch=compute_%s,code=\\\"sm_%s,compute_%s\\\" ', ...
              arch_code, arch_code, arch_code) ;
catch
  if opts.verbose 
    fprintf(['%s:\tCUDA: ',...
      'cannot determine the capabilities of the installed GPU;' ...
      'falling back to default\n'], mfilename);
  end
  cudaArch = opts.defCudaArch;
end

