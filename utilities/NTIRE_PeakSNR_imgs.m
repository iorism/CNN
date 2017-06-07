function res = NTIRE_PeakSNR_imgs(F, G, scale)
% NTIRE 2017 image super-resolution challenge scoring function
%
% F - original image
% G - distorted image
% scale factor - determins the number of boundary pixels to ignore (6+scale)
%
% returns res, the PSNR over all pixel values

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

if max(F(:)) > 1    
    F = im2double(F);
end
if max(G(:)) > 1    
    G = im2double(G);
end
E = F - G; % error signal
N = numel(E); % Assume the original signal is at peak (|F|=1)
res = 10*log10( N / sum(E(:).^2) );