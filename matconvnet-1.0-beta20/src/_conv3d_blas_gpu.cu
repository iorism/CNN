#include "cuda_runtime.h"
#include "_conv3d_blas_gpu.h"
#include "_cu_helper.cuh"
#include "staticMem.h"

#include "logmsg.h"
#ifdef TM
#include "Timer.h"
#endif // TM

namespace {
  //// Impl of copying data back and forth for Vol and Convmat
  typedef conv3d_blas_gpu::CpyVolConvmatImpl CpyImpl;

  const int DIR_VOL_TO_CONVMAT   = 0; // nvcc does not support enum instantiation?
  const int DIR_VOL_FROM_CONVMAT = 1;

  __device__ void get_subY (CpyImpl &ip, int ind, int subY[3]) {
    subY[2] = ind / ip.Y.szProd[1];
    ind %= ip.Y.szProd[1];

    subY[1] = ind / ip.Y.szProd[0];
    ind %= ip.Y.szProd[0];

    subY[0] = ind;
  }

  __device__ void get_win_offset3 (CpyImpl &ip, int h_covnmat,  int win_offset[3]) {
     int subY[3];
     get_subY(ip, h_covnmat, subY);

     win_offset[0] = subY[0]*ip.stride[0] - ip.pad[0];
     win_offset[1] = subY[1]*ip.stride[1] - ip.pad[2];
     win_offset[2] = subY[2]*ip.stride[2] - ip.pad[4];
  }

  __device__ void offset_to_sub (int offset[3], int i, int j, int k,  int sub[3]) {
    sub[0] = offset[0] + i;
    sub[1] = offset[1] + j;
    sub[2] = offset[2] + k;
  }

  __device__ bool is_in_range (CpyImpl &ip, int vol_sub[3]) {
    return ( vol_sub[0] >= 0  &&  vol_sub[0] < ip.X.sz[0]  &&
             vol_sub[1] >= 0  &&  vol_sub[1] < ip.X.sz[1]  &&
             vol_sub[2] >= 0  &&  vol_sub[2] < ip.X.sz[2] );
  }

  template<int dir>
  void __global__ kernelCpyVolConvmat (CpyImpl ip) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < ip.YP) {
      
      // get filter window offset on volume
      int win_offset[3];
      int h_convmat = ind % ip.Y.szProd[2];
      get_win_offset3(ip, h_convmat,  win_offset);

      // get the beginning of 3D volume: vol4d(:,:,:, p)
      int p = ind / ip.Y.szProd[2];
      float* vol3d_beg = ip.vol4d_beg + p * ip.X.szProd[2];

      // get the pointer to convmat  
      float* p_convmat = ip.convmat_beg + (p * ip.Y.szProd[2] * ip.F.szProd[2]) + h_convmat; 

      // scan the sub volume (size F) and copy data
      for (int k = 0; k < ip.F.sz[2]; ++k) {
        for (int j = 0; j < ip.F.sz[1]; ++j) {
          for (int i = 0; i < ip.F.sz[0]; ++i) {
            int vol3d_subscript[3];
            offset_to_sub(win_offset, i,j,k, vol3d_subscript);

            if ( is_in_range (ip, vol3d_subscript) ) { // in range
              // get pointer to vol3d
              float *p_vol3d = vol3d_beg + vol3d_subscript[0] + 
                                           vol3d_subscript[1]*ip.X.szProd[0] + 
                                           vol3d_subscript[2]*ip.X.szProd[1]; 

              if (dir == DIR_VOL_TO_CONVMAT)
                *p_convmat = *p_vol3d;
              else
                atomicAdd(p_vol3d, *p_convmat); // *p_vol3d += p_convmat
            } 
            else { // out of range
              if (dir == DIR_VOL_TO_CONVMAT)
                *p_convmat = 0.0;
              // else: do nothing
            }

            // advance to next convmat element
            p_convmat += ip.Y.szProd[2];
          } // for i
        } // for j
      } // for k

    } // if (ind < ip.YP)
  }

} // namespace


//// impl of public methods
conv3d_blas_gpu::conv3d_blas_gpu()
{

}

conv3d_blas_gpu::conv3d_blas_gpu(const conv3d& obj)
{
  for (int i = 0; i < 6; ++i) pad[i]  = obj.pad[i];
  for (int i = 0; i < 3; ++i) stride[i] = obj.stride[i];

  X  = obj.X;
  dX = obj.dX;
  Y  = obj.Y;
  dY = obj.dY;
  F  = obj.F;
  dF = obj.dF;
  B  = obj.B;
  dB = obj.dB;

  ct = obj.ct;
}

void conv3d_blas_gpu::fprop()
{
  create_Y();
  initStaticMem_convmat();
  initStaticMem_u(); 

#ifdef TM
  Timer tm;
  tm.start();
#endif // TM

  try {
    // iterate over each training instance
    CpyVolConvmatImpl ip = make_initial_CpyVolConvmatImpl( X );
    matw F_ = make_F_();
    matw B_ = make_B_();
    mwSize N = X.getSizeAtDim(4);
    for (mwSize i = 0; i < N; i++) {
      // make phiX: the convolution matrix
      vol_to_convmat(ip, X, i);

      // convolution: Y_ = phiX * F_
      matw Y_ = make_Y_(i);
      cu_AxBtoC(convmat, F_, Y_, true); // overwrite Y_ 

      // plus the bias: Y_ += u * B
      cu_AxBtoC(u, B_, Y_, false); // accumulation on Y_
    } // for i
  } // try
  catch (const blas_ex& e) {
    throw conv3d_ex(e.what());
  }
  catch (const sm_ex& e) {
    throw conv3d_ex(e.what());
  }

#ifdef TM
  cudaThreadSynchronize();

  tm.stop();
  double te = tm.getElapsedTimeInMilliSec();

  mexPrintf("conv3d_blas_gpu::fprop: %f\n", te);
#endif // TM

}

void conv3d_blas_gpu::bprop()
{
  check_X_size();
  create_dX();
  create_dF();
  create_dB();
  initStaticMem_convmat();
  initStaticMem_u();

  try {
    // iterate over each instance
    CpyVolConvmatImpl ip = make_initial_CpyVolConvmatImpl( X );
    matw dF_ = make_dF_();
    matw F_  = make_F_();
    matw dB_ = make_dB_();
    mwSize N = X.getSizeAtDim(4);
    for (mwSize i = 0; i < N; ++i) {
      // make phiX: the convolution matrix
      vol_to_convmat(ip, X, i);

      // dF += phiX' * dY_
      matw dY_ = make_dY_(i);
      cu_ATxBtoC(convmat, dY_, dF_, false); // accumulation on dF_ 

      // dB += u' * dY
      cu_ATxBtoC(u, dY_, dB_, false); // accumulation on dB_

      // dphiX = dY * F'
      // safe to reuse convmat memory since X and dX have the same size; remember to overwrite it!
      cu_AxBTtoC(dY_, F_, convmat, true);
      // dX(:,:,:,:,i) <-- dphiX
      vol_from_convmat(ip, dX, i);
    }
  }
  catch (const blas_ex& e) {
    throw conv3d_ex(e.what());
  }
  catch (const sm_ex& e) {
    throw conv3d_ex(e.what());
  }

}

//// Impl of helper: fprop
matw conv3d_blas_gpu::make_F_()
{
  matw F_;
  F_.beg = (float*)F.getDataBeg();
  F_.H   = numelVol(F) * F.getSizeAtDim(3);
  F_.W   = F.getSizeAtDim(4);

  return F_;
}

matw conv3d_blas_gpu::make_Y_(mwSize i)
{
  matw Y_;
  Y_.beg = getVolInstDataBeg<float>(Y, i);
  Y_.H   = numelVol(Y);
  Y_.W   = Y.getSizeAtDim(3);

  return Y_;
}

matw conv3d_blas_gpu::make_B_()
{
  matw B_;
  B_.beg = (float*)B.getDataBeg();
  B_.H   = 1;
  B_.W   = numel(B);

  return B_;
}

//// Impl of helper: bprop
matw conv3d_blas_gpu::make_dY_(mwSize i)
{
  matw dY_;
  dY_.beg = getVolInstDataBeg<float>(dY, i);
  dY_.H   = numelVol(dY);
  dY_.W   = dY.getSizeAtDim(3);

  return dY_;
}

matw conv3d_blas_gpu::make_dF_()
{
  matw dF_;
  dF_.beg = (float*)dF.getDataBeg();
  dF_.H   = numelVol(dF) * dF.getSizeAtDim(3);
  dF_.W   = dF.getSizeAtDim(4);

  return dF_;
}

matw conv3d_blas_gpu::make_dB_()
{
  matw dB_;
  dB_.beg = (float*)dB.getDataBeg();
  dB_.H   = 1;
  dB_.W   = numel(dB);

  return dB_;
}

//// Impl of helper: the stacked matrix storing phiX or dphiX
conv3d_blas_gpu::CpyVolConvmatImpl conv3d_blas_gpu::make_initial_CpyVolConvmatImpl(const xpuMxArrayTW &vol)
{
  CpyVolConvmatImpl ip;

  // Source, Target data
  ip.vol4d_beg = 0; // set later
  ip.convmat_beg = convmat.beg;

  // volume size for X(or dX), F, Y (or dY)
  ip.X.set_sz(vol);
  ip.F.set_sz(F);
  ip.Y.set_sz( (Y.pa_cpu != 0) ? (Y) : (dY) );
  
  // index and max size for input feature map
  ip.P = vol.getSizeAtDim(3);

  // pre-computation: should always be Y.szProd[2]*P
  ip.YP = ip.Y.szProd[2] * ip.P;

  // other information
  for (int i = 0; i < 3; ++i) ip.stride[i] = (int)this->stride[i];
  for (int i = 0; i < 6; i++) ip.pad[i] = (int)this->pad[i];

  return ip;
}

void conv3d_blas_gpu::initStaticMem_convmat()
{
  // set the size
  assert( (Y.pa_cpu != 0) || (dY.pa_cpu != 0) );
  convmat.H = (Y.pa_cpu != 0) ? numelVol(Y) : numelVol(dY); // FPROP, Y; BPROP, dY
  convmat.W = numelVol(F) * F.getSizeAtDim(3);
  mwSize nelem = convmat.H * convmat.W;

  // the mem
  convmat.beg = sm_zeros(nelem, X.getDevice()); // X always exists

  LOGMSG("conv3d_blas_gpu::initStaticMem_convmat(): %d KB\n", toKB(nelem, mxSINGLE_CLASS));
}

void conv3d_blas_gpu::vol_to_convmat (CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst)
{
  // set vol(:,:,:,:, i)
  ip.vol4d_beg = getVolInstDataBeg<float>(vol, iInst);

  // do the real job
  dim3 blkSize( ceil_divide(ip.YP, CU_NUM_THREADS) );
  dim3 thdSize( CU_NUM_THREADS );
  kernelCpyVolConvmat<DIR_VOL_TO_CONVMAT><<<blkSize, thdSize>>>(ip);
}

void conv3d_blas_gpu::vol_from_convmat(CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst)
{
  // set vol(:,:,:,:, i)
  ip.vol4d_beg = getVolInstDataBeg<float>(vol, iInst);

  // do the real job
  dim3 blkSize( ceil_divide(ip.YP, CU_NUM_THREADS) );
  dim3 thdSize( CU_NUM_THREADS );
  kernelCpyVolConvmat<DIR_VOL_FROM_CONVMAT><<<blkSize, thdSize>>>(ip);
}

void conv3d_blas_gpu::initStaticMem_u()
{
  // decide the size
  assert( (Y.pa_cpu != 0) || (dY.pa_cpu != 0) );
  u.H = (Y.pa_cpu != 0) ? numelVol(Y) : numelVol(dY); // FPROP, Y; BPROP, dY
  u.W = 1;
  mwSize nelem = u.H * u.W ;

  // allocate the memory
  u.beg = sm_ones(nelem, X.getDevice()); // X always exists

  LOGMSG("conv3d_blas_gpu::initStaticMem_u(): %d KB\n", toKB(nelem, mxSINGLE_CLASS));
}
