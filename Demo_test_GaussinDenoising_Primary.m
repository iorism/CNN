%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Made by 'Woong.Bae' (iorism@kaist.ac.kr) at 2017.4.16
% CVPRW 2017 Paper : Beyond Deep Residual Learning for Image Restoration: Persistent Homology-Guided Manifold Simplification


% Copyright <2017> <Woong.Bae(iorism@kaist.ac.kr)>
% 
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
% OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

clear;
% close all;

g=gpuDevice(1);
reset(g); %GPU reset

%% Path setting
addpath('utilities');
addpath('matconvnet-1.0-beta20\matlab');  %%% input matconvnet path
addpath('matconvnet-1.0-beta20\matlab\simplenn');  %%% input matconvnet path
run('vl_setupnn.m');
% run(fullfile('matconvnet-1.0-beta20\matlab\simplenn\vl_simplenn.m'));
%% testing setting
imageSets   = {'Set12'};  %%% select the datasets for each tasks
% Set12 , BSD68

noiseSigma  = 30;
load(fullfile('model_Primary',['Denoising_G',num2str(noiseSigma,'%02d'),'.mat']));
% load(fullfile('model_Primary','\Denoising_Blind.mat')); %%% learned dataset for Primary

bFileSave = 0;  % whether to save the resulting image or not
bPatchMode = 1; % For large image

setTest     = {imageSets([1])};
folderTest  = 'testsets';
folderlable  = 'lablesets';
folderResult = 'results';

%% start SISR
elapsed_time_Total = 0;           

if ~exist(folderResult,'file')
    mkdir(folderResult);
end

st = dwtmode('sym'); %sym %ppd
net.layers(end) = [] ;
net = vl_simplenn_move(net, 'gpu') ;   

setTestCur = cell2mat(setTest{1}(1));
disp('--------------------------------------------');
disp(['----',setTestCur,'-----Gaussian Denosiing-----']);
disp('--------------------------------------------');
folderTestCur = fullfile(folderTest,setTestCur);
folderLableCur = fullfile(folderlable,setTestCur);
ext                 =  {'*.jpg','*.png','*.bmp'};
filepaths_Low           =  [];
filepaths_Lable           =  [];
for i = 1 : length(ext)
    filepaths_Low = cat(1,filepaths_Low,dir(fullfile(folderTestCur, ext{i})));
    filepaths_Lable = cat(1,filepaths_Lable,dir(fullfile(folderLableCur, ext{i})));            
end

%%% folder to store results
folderResultCur = fullfile(folderResult, ['Denosiing','_',setTestCur]);
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end

PSNRs_1 = zeros(1,length(filepaths_Low));
SSIMs_1 = zeros(1,length(filepaths_Low)); 
for i = 1 : length(filepaths_Low)
   label  = imread(fullfile(folderTestCur,filepaths_Low(i).name));
   [~,imageName,ext] = fileparts(filepaths_Low(i).name);                          
   chanel = size(label,3);          
   if chanel == 3                  
        LR_bic_ycc = rgb2ycbcr(label);        
        input_ORG = single(LR_bic_ycc(:,:,1));
   else        
        input_ORG = single(label);
   end
   input_ORG        = double(label) + noiseSigma*randn(size(label));       
      %% Restoration
        tic;                     
        Ysize = ceil(size(input_ORG,1)/2); % +3;
        Xsize = ceil(size(input_ORG,2)/2); % +3;        
        input = zeros(Ysize,Xsize,4,'single');
        [input(:,:,1), input(:,:,2), input(:,:,3), input(:,:,4)] = dwt2(double(input_ORG), 'haar');

         ImageSize = Ysize*Xsize;
         if bPatchMode == 1 && ImageSize > 144864
            patchmode = 0;
            if ImageSize > 500259  && ImageSize < 750000
                patchmode = 1;
            elseif ImageSize >= 750000
                patchmode = 2;
            end
            output_T = double(runPatchWNet(net, input, 1, 20, patchmode));
        else
            input = gpuArray(input);
            res    = vl_simplenn(net,input,[],[],1,'conserveMemory',true,'mode','test');                     
            output = input - res(end).x;
            output_T = double(gather(output));
         end                                
                    
        output = idwt2(output_T(:,:,1),output_T(:,:,2),output_T(:,:,3),output_T(:,:,4),'haar');         
%                     output = vl_nnsubpixelt(output_T, dsT, dsT);    
        toc;

        %%% output (single)
%         if chanel == 3
%             %%% output_RGB (uint8)
%             LR_bic_ycc(:,:,1) = double(output/255.0);
%             output = ycbcr2rgb( uint8(LR_bic_ycc*255) ); %output = double(ycbcr2rgb(LR_bic_ycc))*255;   %im2uint8
%         else
            %%% output_RGB (uint8)
            output = uint8(output); %output = double(output); %im2uint8(output/255.0);
%         end

        [PSNRs_1(i),SSIMs_1(i)] = compute_psnr(label,output,0,0);
        disp(['Gaussian Denoising  ',num2str(PSNRs_1(i),'%2.4f'),'dB',num2str(SSIMs_1(i),'%2.4f')]);                            
        toc;
        EachTime = toc;
        elapsed_time_Total = elapsed_time_Total + EachTime;                                               
        %% save results
        if bFileSave == 1
            imwrite(output,fullfile(folderResultCur,[num2str(i,'%04d'),'.png']));
        end
end           
disp([mean(PSNRs_1),mean(SSIMs_1)]);
elapsed_time_Total = elapsed_time_Total / length(filepaths_Low);
disp(['Total Time : ',num2str(elapsed_time_Total,'%4.4f'),'sec']);                





