function y = vl_nnsubpixel(x, dy, dx, dzdx)

[ny, nx, nch, nout] = size(x);

% dy                  = dy;
% dx                  = dy;

my                  = ny/dy;
mx                  = nx/dx;
mch                 = nch*dy*dx;

iy                  = 1:dy:ny;
ix                  = 1:dx:nx;


if nargin == 3
    
    y           	= zeros(my, mx, mch, nout, 'like', x);
    
    for j = 0:dy-1
        for i = 0:dx-1
            k               = i + dx*j + (1:nch);
            y(:, :, k, :)	= x(iy+j, ix+i, :, :);
        end
    end
    
else
    
    y               = zeros(ny, nx, nch, nout, 'like', x);
    
    for j = 0:dy-1
        for i = 0:dx-1
            k                   = j + dy*i + (1:nch);
            y(iy+j, ix+i, :, :)	= dzdx(:, :, k, :);
        end
    end
end