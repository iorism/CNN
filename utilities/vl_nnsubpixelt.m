function y = vl_nnsubpixelt(x, dy, dx, dzdx)

[my, mx, mch, nout]	= size(x);

% dy                  = dy;
% dx                  = dy;

ny                  = my*dy;
nx                  = mx*dx;
nch                 = mch/dy/dx;

iy                  = 1:dy:ny;
ix                  = 1:dx:nx;


if nargin == 3
    y               = zeros(ny, nx, nch, nout, 'like', x);
    
    for j = 0:dy-1
        for i = 0:dx-1
            k               = i + dx*j + (1:nch);
            y(iy+j, ix+i, :, :)	= x(:, :, k, :);
        end
    end    

    
else
    y           	= zeros(my, mx, mch, nout, 'like', x);
    
    for j = 0:dy-1
        for i = 0:dx-1
            k               = j + dy*i + (1:nch);
            y(:, :, k, :)	= dzdx(iy+j, ix+i, :, :);
        end
    end
end