#include "cuda_runtime.h"
#include "_conv3d_blas_gpu.h"
#include "_cu_helper.h"
#include "logmsg.h"
#ifdef TM
#include "Timer.h"
#include <cublas_v2.h>
#endif // TM

namespace {
//// helpers for threads


//// helper: setting initial value
template<typename T>
__global__ void kernelSetZero (T* beg, mwSize len) {
  mwSize ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < len) beg[ind] = static_cast<T>(0);
}

template<typename T>
__global__ void kernelSetOne (T* beg, mwSize len) {
  mwSize ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < len) beg[ind] = static_cast<T>(1);
}

//// Impl of copying data back and forth for Vol and Convmat
typedef conv3d_blas_gpu::CpyVolConvmatImpl CpyImpl;

__device__ mwSize get_convmat_h (CpyImpl &ip, mwSize indCM) {
  return (indCM % ip.convmat.H);
}

__device__ mwSize get_convmat_w (CpyImpl &ip, mwSize indCM) {
  return (indCM / ip.convmat.H);
}

__device__ void get_subY (CpyImpl &ip, mwSize ind,  mwSize subY[3]) {
  mwSize HW = ip.szY[0] * ip.szY[1];
  mwSize H  = ip.szY[0];

  subY[2] = ind / HW;
  ind %= HW;

  subY[1] = ind / H;
  ind %= H;

  subY[0] = ind;
}

__device__ void get_win_offset4 (CpyImpl &ip, mwSize h_covnmat,  int64_T win_offset[3]) {
  mwSize subY[3];
  get_subY(ip, h_covnmat, subY);

  for (int i = 0; i < 3; ++i) 
    win_offset[i] = -static_cast<int64_T>(ip.pad[2*i]) + static_cast<int64_T>(subY[i] * ip.stride[i]);
  win_offset[3] = 0;
}

__device__ void get_win_sub4 (CpyImpl &ip, mwSize w_convmat,  mwSize win_sub[4]) {
  mwSize H   = ip.szF[0]; 
  mwSize HW  = H * ip.szF[1];
  mwSize HWD = HW * ip.szF[2];

  win_sub[3] = w_convmat / HWD;
  w_convmat %= HWD;

  win_sub[2] = w_convmat / HW;
  w_convmat %= HW;

  win_sub[1] = w_convmat / H;
  w_convmat %= H;

  win_sub[0] = w_convmat;
}

// return -1 if out of range (either underflow or overflow)
__device__ int64_T get_indVol (CpyImpl &ip, int64_T win_offset[4], mwSize win_sub[4]) {

  // the global subscript and guaranteed valid range
  int64_T vol_sub[4];
  for (int i = 0; i < 4; ++i) {
    vol_sub[i] = win_offset[i] + static_cast<int64_T>(win_sub[i]);
    if ( vol_sub[i] < 0 ) return -1; // underflow
    if ( vol_sub[i] >= ip.vol_i.sz[i] ) return -1; // overflow
  }

  mwSize H   = ip.vol_i.sz[0];
  mwSize HW  = H * ip.vol_i.sz[1];
  mwSize HWD = HW * ip.vol_i.sz[2];

  return static_cast<int64_T>(HWD*vol_sub[3] + HW*vol_sub[2] + H*vol_sub[1] + vol_sub[0]);
}

const int DIR_VOL_TO_CONVMAT   = 0; // nvcc does not support enum instantiation?
const int DIR_VOL_FROM_CONVMAT = 1;

template<int dir>
void __global__ kernelCpyVolConvmat (CpyImpl ip) {
  mwSize indCM = blockDim.x * blockIdx.x + threadIdx.x;
  if ( indCM >= (ip.convmat.H*ip.convmat.W) ) return;

  // fill h, w
  mwSize h = get_convmat_h(ip, indCM); // convmat dim1
  mwSize w = get_convmat_w(ip, indCM); // convmat dim2

  // h (convmat dim1) -> window's offset (starting point) on volume (win_offset[3] = 0 as volume dim4 all in!)
  int64_T win_offset[4]; // fill win_offset
  get_win_offset4(ip, h, win_offset);

  // w (convmat dim2) -> win_sub ( r,s,t,u the subscript within the window )
  mwSize win_sub[4]; // (r, s, t, u) 
  get_win_sub4(ip, w, win_sub);

  // win_offset[4] and win_sub[4] -> linear index, ind, on volume
  int64_T indVol = get_indVol(ip, win_offset, win_sub);
  
  // copy the data at indCM, indVol
  if (indVol < 0) {
    if (dir == DIR_VOL_TO_CONVMAT) 
      ip.convmat.beg[indCM] = 0.0; // pad zeros!
    //else: DIR_VOL_FROM_CONVMAT, do nothing
    return;
  }
  
  if (dir == DIR_VOL_TO_CONVMAT) // vol -> convmat
    ip.convmat.beg[indCM] = ip.vol_i.beg[indVol];
  else { // DIR_VOL_FROM_CONVMAT, vol <- convmat
    // ATOMIC increment: ip.vol_i.beg[indVol] += ip.convmat.beg[indCM]
    atomicAdd( (ip.vol_i.beg + indVol), ip.convmat.beg[indCM]);
  }
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
  init_convmat();

#ifdef TM
  Timer tm;
#endif // TM

  try {
    // iterate over each training instance
    CpyVolConvmatImpl ip = make_initial_CpyVolConvmatImpl( X );
    mwSize N = X.getSizeAtDim(4);

#ifdef TM
    mexPrintf("num inst = %d \n", N);
    tm.start();
#endif

    for (mwSize i = 0; i < N; i++) {

      // make phiX: the convolution matrix
      vol_to_convmat(ip, X, i);
      

    } // for i
  } // try
  catch (const blas_ex& e) {
    free_convmat();
    throw conv3d_ex(e.what());
  }

#ifdef TM
  cudaThreadSynchronize();

  tm.stop();
  double te = tm.getElapsedTimeInMilliSec();

  mexPrintf("conv3d_blas_gpu::vol_to_convmat: %f\n", te);
#endif // TM

  free_convmat();
  //free_u();

#ifdef TM
  mexPrintf("\n");
#endif // TM
}

//void conv3d_blas_gpu::fprop()
//{
//  create_Y();
//  init_convmat();
//  init_u(); 
//
//#ifdef TM
//  Timer tm;
//  tm.start();
//
//  Timer tmtm;
//  double t1 = 0.0, t2 = 0.0, t3 = 0.0;
//  mwSize H1,W1, H2,W2;
//  mwSize H3,W3, H4,W4;
//#endif // TM
//
//  try {
//    // iterate over each training instance
//    CpyVolConvmatImpl ip = make_initial_CpyVolConvmatImpl( X );
//    mwSize N = X.getSizeAtDim(4);
//
//#ifdef TM
//    mexPrintf("num inst = %d \n", N);
//#endif
//
//    for (mwSize i = 0; i < N; i++) {
//#ifdef TM
//      tmtm.start();
//#endif // TM
//
//      // make phiX: the convolution matrix
//      vol_to_convmat(ip, X, i);
//
//#ifdef TM
//      tmtm.stop();
//      t1 += tmtm.getElapsedTimeInMilliSec();
//#endif // TM
//
//#ifdef TM
//      tmtm.start();
//#endif // TM
//
//      // convolution: Y_ = phiX * F_
//      matw F_ = make_F_();
//      matw Y_ = make_Y_(i);
//      //cu_AxBtoC(convmat, F_, Y_, true); // overwrite Y_ 
//
//#ifdef TM
//      tmtm.stop();
//      t2 += tmtm.getElapsedTimeInMilliSec();
//      H1 = convmat.H; W1 = convmat.W;
//      H2 = F_.H; W2 = F_.W;
//#endif // TM
//
//#ifdef TM
//      tmtm.start();
//#endif // TM
//
//      // plus the bias: Y_ += u * B
//      matw B_ = make_B_();
//      //cu_AxBtoC(u, B_, Y_, false); // accumulation on Y_
//
//#ifdef TM
//      tmtm.stop();
//      t3 += tmtm.getElapsedTimeInMilliSec();
//      H3 =  u.H; W3 = u.W;
//      H4 = B_.H; W4 = B_.W;
//#endif // TM
//    } // for i
//  } // try
//  catch (const blas_ex& e) {
//    free_u();
//    free_convmat();
//    throw conv3d_ex(e.what());
//  }
//
//#ifdef TM
//  tm.stop();
//  double te = tm.getElapsedTimeInMilliSec();
//
//  mexPrintf("conv3d_blas_gpu::vol_to_convmat: %f\n", t1);
//
//  mexPrintf("conv3d_blas_gpu::mut 1: %f\n", t2);
//  mexPrintf("[%d %d] x [%d %d]\n", H1,W1, H2,W2);
//
//  mexPrintf("conv3d_blas_gpu::mut 2: %f\n", t3);
//  mexPrintf("[%d %d] x [%d %d]\n",H3,W3, H4,W4);
//
//  mexPrintf("conv3d_blas_gpu::fprop: %f\n", te);
//#endif // TM
//
//  //free_convmat();
//  //free_u();
//  
//#ifdef TM
//  mexPrintf("\n");
//#endif // TM
//}

//void conv3d_blas_gpu::fprop()
//{
//  create_Y();
//  init_convmat();
//  init_u(); 
//
//#ifdef TM
//  Timer tm;
//  tm.start();
//
//  Timer tmtm;
//  double t1 = 0.0, t2 = 0.0, t3 = 0.0;
//  mwSize H1,W1, H2,W2;
//  mwSize H3,W3, H4,W4;
//
//  cublasHandle_t hd;
//  cublasStatus_t st;
//  cublasCreate(&hd);
//
//#endif // TM
//
//  try {
//    // iterate over each training instance
//    CpyVolConvmatImpl ip = make_initial_CpyVolConvmatImpl( X );
//    mwSize N = X.getSizeAtDim(4);
//
//#ifdef TM
//    mexPrintf("num inst = %d \n", N);
//#endif
//
//    for (mwSize i = 0; i < N; i++) {
//#ifdef TM
//      tmtm.start();
//#endif // TM
//
//      // make phiX: the convolution matrix
//      //vol_to_convmat(ip, X, i);
//
//#ifdef TM
//      tmtm.stop();
//      t1 += tmtm.getElapsedTimeInMilliSec();
//#endif // TM
//
//
//      // convolution: Y_ = phiX * F_
//      matw F_ = make_F_();
//      matw Y_ = make_Y_(i);
//
//      float alpha = 1.0;
//      float beta = 0.0;
//
//#ifdef TM
//      tmtm.start();
//#endif // TM
//
//      st = cublasSgemm(
//        hd,
//        CUBLAS_OP_N, CUBLAS_OP_N,
//        (int)convmat.H, (int)F_.W, (int)convmat.W,
//        &alpha,
//        (float*)convmat.beg, (int)convmat.H,
//        (float*)F_.beg, (int)F_.H,
//        &beta,
//        (float*)Y_.beg, (int)Y_.H);
//
//#ifdef TM
//      tmtm.stop();
//      t2 += tmtm.getElapsedTimeInMilliSec();
//      H1 = convmat.H; W1 = convmat.W;
//      H2 = F_.H; W2 = F_.W;
//#endif // TM
//
//
//      // plus the bias: Y_ += u * B
//      matw B_ = make_B_();
//      
//#ifdef TM
//      tmtm.start();
//#endif // TM
//
//      alpha = 1.0;
//      beta  = 1.0;
//      st = cublasSgemm(
//        hd,
//        CUBLAS_OP_N, CUBLAS_OP_N,
//        (int)u.H, (int)B_.W, (int)u.W,
//        &alpha,
//        (float*)u.beg, (int)u.H,
//        (float*)B_.beg, (int)B_.H,
//        &beta,
//        (float*)Y_.beg, (int)Y_.H);
//
//#ifdef TM
//      tmtm.stop();
//      t3 += tmtm.getElapsedTimeInMilliSec();
//      H3 =  u.H; W3 = u.W;
//      H4 = B_.H; W4 = B_.W;
//#endif // TM
//    } // for i
//  } // try
//  catch (const blas_ex& e) {
//    free_u();
//    free_convmat();
//    throw conv3d_ex(e.what());
//  }
//
//#ifdef TM
//  cublasDestroy(hd);
//  tm.stop();
//  double te = tm.getElapsedTimeInMilliSec();
//
//  mexPrintf("conv3d_blas_gpu::vol_to_convmat: %f\n", t1);
//
//  mexPrintf("conv3d_blas_gpu::mut 1: %f\n", t2);
//  mexPrintf("[%d %d] x [%d %d]\n", H1,W1, H2,W2);
//
//  mexPrintf("conv3d_blas_gpu::mut 2: %f\n", t3);
//  mexPrintf("[%d %d] x [%d %d]\n",H3,W3, H4,W4);
//
//  mexPrintf("conv3d_blas_gpu::fprop: %f\n", te);
//#endif // TM
//
//  free_convmat();
//  free_u();
//
//#ifdef TM
//  mexPrintf("\n");
//#endif // TM
//
//}

void conv3d_blas_gpu::bprop()
{
  check_X_size();
  create_dX();
  create_dF();
  create_dB();
  init_convmat();
  init_u();

  try {
    // iterate over each instance
    CpyVolConvmatImpl ip = make_initial_CpyVolConvmatImpl( X );
    matw dF_ = make_dF_();
    matw dB_ = make_dB_();
    mwSize N = X.getSizeAtDim(4);
    for (mwSize i = 0; i < N; ++i) {
      // make phiX: the convolution matrix
      vol_to_convmat(ip, X, i);

      // dF += phiX' * dY_
      matw dY_ = make_dY_(i);
      cu_ATxBtoC(convmat, dY_, dF_, false); // accumulation on dF_ TODO: the right cublas

      // dB += u' * dY
      cu_ATxBtoC(u, dY_, dB_, false); // accumulation on dB_

      // dphiX = dY * F'
      matw F_ = make_F_();
      // safe to reuse convmat memory as X and dX have the same size; remember to overwrite it!
      cu_AxBTtoC(dY_, F_, convmat, true);
      // dX(:,:,:,:,i) <-- dphiX
      vol_from_convmat(ip, dX, i);
    }
  }
  catch (const blas_ex& e) {
    free_u();
    free_convmat();
    throw conv3d_ex(e.what());
  }

  free_u();
  free_convmat();
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

  ip.vol_i.beg = 0; // to be set later
  for (int i = 0; i < 4; ++i) ip.vol_i.sz[i] = vol.getSizeAtDim(i);

  ip.convmat = this->convmat;

  if ( Y.pa_cpu != 0)
    for (int i = 0; i < 3; ++i) ip.szY[i] = this->Y.getSizeAtDim(i);
  else // dY.pa_cpu != 0
    for (int i = 0; i < 3; ++i) ip.szY[i] = this->dY.getSizeAtDim(i);

  for (int i = 0; i < 3; ++i) ip.szF[i] = this->F.getSizeAtDim(i);
  for (int i = 0; i < 3; ++i) ip.stride[i] = this->stride[i];
  for (int i = 0; i < 6; i++) ip.pad[i] = this->pad[i];

  return ip;
}

void conv3d_blas_gpu::init_convmat()
{
#ifdef TM
  Timer tm;
  tm.start();
#endif // TM

  // set the size
  assert( (Y.pa_cpu != 0) || (dY.pa_cpu != 0) );
  if (Y.pa_cpu != 0) // in FPROP, Y has been set
    convmat.H = numelVol(Y);
  else // (dY != 0), in BPROP, dY has been set
    convmat.H = numelVol(dY);

  convmat.W = numelVol(F) * F.getSizeAtDim(3);
  mwSize nelem = convmat.H * convmat.W;

  // allocate the memory
  void* tmp;
  cudaError_t flag = cudaMalloc(&tmp,  nelem*sizeof(float) ) ;
  if (flag != cudaSuccess) throw conv3d_ex("Out of memory on GPU.\n");
  convmat.beg = (float*)tmp;

  // assures all zeros
  kernelSetZero<float><<<ceil_divide(nelem,CU_NUM_THREADS), CU_NUM_THREADS>>>(convmat.beg, nelem);
  
#ifdef TM
  tm.stop();
  double te = tm.getElapsedTimeInMilliSec();
  mexPrintf("conv3d_blas_gpu::init_convmat: %f\n", te);
#endif // TM

  LOGMSG("conv3d_blas_gpu::init_convmat(): %d KB\n", toKB(nelem, mxSINGLE_CLASS));
}

void conv3d_blas_gpu::free_convmat()
{
//#ifdef TM
//  Timer tm;
//  tm.start();
//#endif // TM

  cudaFree( (void*)convmat.beg );

//#ifdef TM
//  tm.stop();
//  double te = tm.getElapsedTimeInMilliSec();
//  mexPrintf("conv3d_blas_gpu::free_convmat: %f\n", te);
//#endif // TM

  LOGMSG("conv3d_blas_gpu::free_convmat()\n");
}

void conv3d_blas_gpu::vol_to_convmat (CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst)
{
  // set vol(:,:,:,:, i)
  ip.vol_i.beg = getVolInstDataBeg<float>(vol, iInst);

  // do the real job
  mwSize nelem = ip.convmat.H * ip.convmat.W;
  dim3 blkSize( ceil_divide(nelem, CU_NUM_THREADS) );
  dim3 thdSize( CU_NUM_THREADS );
  kernelCpyVolConvmat<DIR_VOL_TO_CONVMAT><<<blkSize, thdSize>>>(ip);
}

void conv3d_blas_gpu::vol_from_convmat(CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst)
{
  // set vol(:,:,:,:, i)
  ip.vol_i.beg = getVolInstDataBeg<float>(vol, iInst);

  // do the real job
  mwSize nelem = ip.convmat.H * ip.convmat.W;
  dim3 blkSize( ceil_divide(nelem, CU_NUM_THREADS) );
  dim3 thdSize( CU_NUM_THREADS );
  kernelCpyVolConvmat<DIR_VOL_FROM_CONVMAT><<<blkSize, thdSize>>>(ip);
}

void conv3d_blas_gpu::init_u()
{
#ifdef TM
  Timer tm;
  tm.start();
#endif // TM

  // decide the size
  assert( (Y.pa_cpu != 0) || (dY.pa_cpu != 0) );
  if (Y.pa_cpu != 0)
    u.H = numelVol(Y);
  else // (dY != 0)
    u.H = numelVol(dY);

  u.W = 1;
  mwSize nelem = u.H * u.W ;

  // allocate the memory
  void* tmp;
  cudaError_t flag = cudaMalloc(&tmp, nelem * sizeof(float));
  if (flag != cudaSuccess) throw conv3d_ex("Out of memory on GPU.\n");
  u.beg = (float*) tmp;

  // make sure all one
  kernelSetOne<float><<<ceil_divide(nelem,CU_NUM_THREADS), CU_NUM_THREADS>>>(u.beg, nelem);

  LOGMSG("conv3d_blas_gpu::init_u(): %d KB\n", toKB(nelem, mxSINGLE_CLASS));

#ifdef TM
  tm.stop();
  double te = tm.getElapsedTimeInMilliSec();
  mexPrintf("onv3d_blas_gpu::init_u: %f\n", te);
#endif // TM
}

void conv3d_blas_gpu::free_u()
{
#ifdef TM
  Timer tm;
  tm.start();
#endif // TM

  cudaFree( (void*)u.beg );

#ifdef TM
  tm.stop();
  double te = tm.getElapsedTimeInMilliSec();
  mexPrintf("conv3d_blas_gpu::free_u: %f\n", te);
#endif // TM

  LOGMSG("conv3d_blas_gpu::free_u()\n");
}