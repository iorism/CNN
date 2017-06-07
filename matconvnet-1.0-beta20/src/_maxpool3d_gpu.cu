#include "cuda_runtime.h"
#include "_maxpool3d_gpu.h"
#include "_cu_helper.cuh"


namespace {

//// thin wrappers 
struct tw_array5d { 
  float *beg;
  int sz[5];
  int HW, HWD;
  int nelem;
};

template<typename T>
struct tw_vec {
  T* beg;
  int sz;
};

struct fprop_impl {
  tw_array5d  X, Y;
  tw_vec<int> ind;
 
  int pool[3];
  int stride[3];
  int pad[6];
};

struct bprop_impl
{
  tw_vec<float>  dX, dY;
  tw_vec<int> ind;
};

//// kernel Impl
__device__ const float VERY_NEGATIVE_NUM = -1e20;

__device__ void ind2sub (int iElem, int sz[5], 
                         int &h, int &w, int &d, int &iVol) 
{
  int H   = sz[0];
  int HW  = H * sz[1];
  int HWD = HW * sz[2];

  iVol = iElem / HWD;
  iElem = iElem % HWD;

  d = iElem / HW;
  iElem = iElem % HW;

  w = iElem / H;

  h = iElem % H;
}

__global__ void kernel_fprop (fprop_impl impl) {
  int iElem = blockIdx.x * blockDim.x + threadIdx.x;

  if (iElem >= impl.Y.nelem) return;

  // subscript on Y
  int iY, jY, kY, iVol;
  ind2sub(iElem, impl.Y.sz,  iY,jY,kY,iVol);

  // init value for current Y
  float vmax = VERY_NEGATIVE_NUM;
  int   imax = -43.0;

  // set the window on X for current Y element (iElem); note the offset can be negative
  int xH   = impl.X.sz[0];
  int xHW  = impl.X.HW;
  int xHWD = impl.X.HWD;
  int xwin_offset[3];
  xwin_offset[0] = -(impl.pad[0]) + ( iY * impl.stride[0] ); 
  xwin_offset[1] = -(impl.pad[2]) + ( jY * impl.stride[1] );
  xwin_offset[2] = -(impl.pad[4]) + ( kY * impl.stride[2] );
  const float* const xwin_beg = impl.X.beg + 
                                xwin_offset[0] + 
                                xwin_offset[1]*xH + 
                                xwin_offset[2]*xHW +
                                iVol*xHWD;

  // inspect the window at X, get the max value
  for (int t = 0; t < impl.pool[2]; ++t) {     // X window dim3: depth
    int xt = t + xwin_offset[2];
    bool xtInRange = (xt>=0) && (xt<impl.X.sz[2]);

    for (int s = 0; s < impl.pool[1]; ++s) {   // X window dim2: width
      int xs = s + xwin_offset[1];
      bool xsInRange = (xs>=0) && (xs<impl.X.sz[1]);

      for (int r = 0; r < impl.pool[0]; ++r) { // X window dim1: height
        int xr = r + xwin_offset[0];
        bool xrInRange = (xr>=0) && (xr<impl.X.sz[0]);

        // if out of range: never collect the element
        if ( !(xtInRange && xsInRange && xrInRange) )
          continue;

        // collect the element: current x value
        float vx = *(xwin_beg + r + s*xH + t*xHW);
        if (vx >= vmax) { // found new max value?
          vmax = vx;
          imax = double( xr + xs*xH + xt*xHW + iVol*xHWD );
        } // if

      } // r
    } // s
  } // t

  // write to the target
  impl.Y.beg[iElem]   = vmax;
  impl.ind.beg[iElem] = imax + 1; // to Matlab 1-base
}

__global__ void kernel_bprop (bprop_impl impl) {
  int iY = blockIdx.x * blockDim.x + threadIdx.x;

  if (iY >= impl.dY.sz) return;

  int ix = int( impl.ind.beg[iY] );
  ix -= 1;

  // atomic Increment: there can be overlapping ix!
  atomicAdd( (impl.dX.beg + ix), impl.dY.beg[iY] );
}

} // namespace

//// impl of public methods
maxpool3d_gpu::maxpool3d_gpu()
{

}

maxpool3d_gpu::maxpool3d_gpu(const maxpool3d &obj)
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

void maxpool3d_gpu::fprop()
{
  // create output
  create_Y();
  create_ind();


  // set the impl struct and run it
  fprop_impl impl;
  // options
  for (int i = 0; i < 3; ++i) impl.pool[i] = pool[i];
  for (int i = 0; i < 6; ++i) impl.pad[i] = pad[i];
  for (int i = 0; i < 3; ++i) impl.stride[i] = stride[i];
  // input: X, device pointer
  impl.X.beg = (float*) X.getDataBeg();
  for (int i = 0; i < 5; ++i) impl.X.sz[i] = X.getSizeAtDim(i);
  impl.X.HW    = impl.X.sz[0] * impl.X.sz[1];
  impl.X.HWD   = impl.X.HW * impl.X.sz[2];
  impl.X.nelem = static_cast<int>( numel(X) );
  // output: Y, device pointer
  impl.Y.beg = (float*) Y.getDataBeg();
  for (int i = 0; i < 5; ++i) impl.Y.sz[i] = Y.getSizeAtDim(i);
  impl.Y.HW    = impl.Y.sz[0] * impl.Y.sz[1];
  impl.Y.HWD   = impl.Y.HW * impl.Y.sz[2];
  impl.Y.nelem = static_cast<int>( numel(Y) );
  // output: ind, device pointer
  impl.ind.beg = (int*) ind.getDataBeg();
  impl.ind.sz  = static_cast<int>( numel(ind) );


  // run
  int nelem = static_cast<int>( numel(Y) );
  kernel_fprop<<<ceil_divide(nelem, CU_NUM_THREADS), CU_NUM_THREADS>>>( impl );
}

void maxpool3d_gpu::bprop()
{
  // create dX at input port
  check_dY_ind();
  create_dX();


  // set the impl struct
  bprop_impl impl;
  //
  impl.dX.beg = (float*) dX.getDataBeg();
  impl.dX.sz  = static_cast<int>( numel(dX) );
  //
  impl.dY.beg = (float*) dY.getDataBeg();
  impl.dY.sz  = static_cast<int>( numel(dY) );
  //
  impl.ind.beg = (int*) ind.getDataBeg();
  impl.ind.sz  = static_cast<int>( numel(ind) );


  // run
  kernel_bprop <<<ceil_divide(impl.dY.sz, CU_NUM_THREADS), CU_NUM_THREADS>>>( impl );
}

