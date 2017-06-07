%% config
rng(6345, 'twister');


%% test Conv3d
% dg  = @t_c3d.dg_cpu;
% dg  = @t_c3d.dg_gpu;

% t_c3d.tc_1(dg);
% t_c3d.tc_3(dg);
% t_c3d.tc_5(dg);
%% test MaxPool3d
% dg = @t_mp3d.dg_cpu;
dg = @t_mp3d.dg_gpu;

% t_mp3d.tc_1(dg);
% t_mp3d.tc_2(dg);
% t_mp3d.tc_3(dg);
% t_mp3d.tc_4(dg);
% t_mp3d.tc_5(dg);
t_mp3d.tc_6(dg);