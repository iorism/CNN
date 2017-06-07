#include "wrapperMx.h"
#include "_maxpool3d_cpu.h"
#include <omp.h>


namespace {
  const float VERY_NEGATIVE_NUM = -1e20f;
}

//// impl of public methods
maxpool3d_cpu::maxpool3d_cpu()
{

}

maxpool3d_cpu::maxpool3d_cpu(const maxpool3d &obj)
{
  for (int i = 0; i < 6; ++i) pad[i]  = obj.pad[i];
  for (int i = 0; i < 3; ++i) pool[i] = obj.pool[i];
  for (int i = 0; i < 3; ++i) stride[i] = obj.stride[i];

  ind = obj.ind;
  X  = obj.X;
  dX = obj.dX;
  Y  = obj.Y;
  dY = obj.dY;

  ct = obj.ct;
}

void maxpool3d_cpu::fprop()
{
  // create output
  create_Y();
  create_ind();

  // input X size
  int xH = X.getSizeAtDim(0), xW = X.getSizeAtDim(1), xD = X.getSizeAtDim(2);
  int xHW  = xH*xW;
  int xHWD = xH*xW*xD;

  // iterate over Y, record the max value and index
  #pragma omp parallel for
  for (int n = 0; n < numVol(Y); ++n) { // Y dim4, dim5...: along each volume

    float* yy = getVolDataBeg<float>(Y, n);    // output data
    int*   ii = getVolDataBeg<int>(ind, n); // output index

    const float* const xx_beg = getVolDataBeg<float>(X, n); // input data (never change it)

    for (int k = 0; k < Y.getSizeAtDim(2); ++k) {     // Y dim3: along depth
      for (int j = 0; j < Y.getSizeAtDim(1); ++j) {   // Y dim2: along width
        for (int i = 0; i < Y.getSizeAtDim(0); ++i) { // Y dim1: along height

          // init value for current Y
          float vmax = VERY_NEGATIVE_NUM;
          int   imax = -43.0;

          // set the window on X for current Y element (yy): the offset can be negative
          int xwin_offset[3];
          xwin_offset[0] = -static_cast<int>(pad[0]) + static_cast<int>( i*stride[0] ); 
          xwin_offset[1] = -static_cast<int>(pad[2]) + static_cast<int>( j*stride[1] );
          xwin_offset[2] = -static_cast<int>(pad[4]) + static_cast<int>( k*stride[2] );
          const float* const xwin_beg = xx_beg + 
                                        xwin_offset[0] + 
                                        xwin_offset[1]*xH + 
                                        xwin_offset[2]*xHW;

          // inspect the window at X, get the max value
          for (int t = 0; t < pool[2]; ++t) {     // X window dim3: depth
            int xt = t + xwin_offset[2];
            bool xtInRange = (xt>=0) && (xt<xD);

            for (int s = 0; s < pool[1]; ++s) {   // X window dim2: width
              int xs = s + xwin_offset[1];
              bool xsInRange = (xs>=0) && (xs<xW);

              for (int r = 0; r < pool[0]; ++r) { // X window dim1: height
                int xr = r + xwin_offset[0];
                bool xrInRange = (xr>=0) && (xr<xH);

                // if out of range: never collect the element
                if ( !(xtInRange && xsInRange && xrInRange) )
                  continue;

                // collect the element: current x value
                float vx = *(xwin_beg + r + s*xH + t*xHW);
                if (vx >= vmax) { // found new max value?
                  vmax = vx;
                  imax = double( xr + xs*xH + xt*xHW + n*xHWD );
                } // if

              } // r
            } // s
          } // t

          // write back to Y and advance
          *yy++ = vmax;
          *ii++ = imax + 1; // to matlab 1-base
        } // i
      } // j
    } // k

  } // parallel for n

}

void maxpool3d_cpu::bprop()
{
  // create dX at input port
  check_dY_ind();
  create_dX();

  // dX at input port
  float* const dxx = (float*)dX.getDataBeg();
  // dY and index at output port
  float* const dyy = (float*)dY.getDataBeg();
  int*   const ii  = (int*)ind.getDataBeg();

  // iterate over dY, set dX
  int num = static_cast<int>( numel(dY) );
  
  #pragma omp parallel for
  for (int n = 0; n < num; ++n) {
    int ix = ii[n];
    ix -= 1; // matlab 1-base -> C++ 0-base

    // accumulate! there can be overlapping ix!
    #pragma omp atomic
    dxx[ix] += dyy[n];
  }
}
