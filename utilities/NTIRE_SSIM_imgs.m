function [mssim, ssim_map] = NTIRE_SSIM_imgs(F, G, scale)
% NTIRE 2017 image super-resolution challenge scoring function
%
% F - original image
% G - distorted image
% scale factor - determins the number of boundary pixels to ignore (6+scale)
%
% returns mssim, the mean SSIM index over the RGB channels

if ischar(F)
    F = imread(F);
end
if ischar(G)
    G = imread(G);
end

boundarypixels = 0;
if exist('scale','var')
    boundarypixels = 6+scale; 
    F = F(boundarypixels+1:end-boundarypixels,boundarypixels+1:end-boundarypixels,:);
    G = G(boundarypixels+1:end-boundarypixels,boundarypixels+1:end-boundarypixels,:);
end

% the image pixel values should be from range [0..255]
mssim = zeros(1,size(F,3));
for i = 1:size(F,3)
    [mssim(i), ssim_map] = ssim_index(F(:,:,i), G(:,:,i)); %default settings
end

mssim = mean(mssim);