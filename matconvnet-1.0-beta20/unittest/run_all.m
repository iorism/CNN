function run_all ()
%% config
% rng(81192, 'twister');

run_type('gpu');
run_type('cpu');

function run_type (type)
%% test Conv3d
if strcmp(type,'cpu')
  dg = @t_c3d.dg_cpu;
else
  dg = @t_c3d.dg_gpu;
end
run_all_conv3d(dg);
%% test MaxPool3d
if strcmp(type,'cpu')
  dg = @t_mp3d.dg_cpu;
else
  dg = @t_mp3d.dg_gpu;
end
run_all_mp3d(dg);


function run_all_conv3d (dg)
t_c3d.tc_1(dg);
t_c3d.tc_2(dg);
t_c3d.tc_3(dg);
t_c3d.tc_4(dg);
t_c3d.tc_5(dg);

function run_all_mp3d (dg)
t_mp3d.tc_1(dg);
t_mp3d.tc_2(dg);
t_mp3d.tc_3(dg);
t_mp3d.tc_4(dg);
t_mp3d.tc_5(dg);