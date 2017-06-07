#include "conv3d.h"
#include "_conv3d_blas_cpu.h"
#ifdef WITH_GPUARRAY
#include "wrapperBlas.h"
#include "_conv3d_blas_gpu.h"
#include "_conv3d_blas_gpu_fc.h"
#endif // WITH_GPUARRAY

#ifdef TM
#include "Timer.h"
#endif // TM

#include "staticMem.h"


//// Impl of conv3d
const char* conv3d::THE_CMD = 
  "Bad input or output arguments. The right way to call:\n"
  "Y = MEX_CONV3D(X,F,B); forward pass\n"
  "[dX,dF,dB] = MEX_CONV3D(X,F,B, dY); backward pass\n"
  "MEX_CONV3D(..., 'stride',s, 'pad',pad); options\n"
  "All arguments must be single.\n";

conv3d::conv3d()
{
  stride[0] = stride[1] = stride[2] = 1;
  pad[0] = pad[1] = pad[2] = pad[3] = pad[4] = pad[5] = 0;

}

conv3d::~conv3d()
{

}

//// Impl of helpers
void conv3d::create_Y()
{
//#ifdef TM
//  Timer tm;
//  tm.start();
//#endif // TM

  // check input X and filter F, B
  if ( F.getSizeAtDim(3) != X.getSizeAtDim(3) )  // 
    throw conv3d_ex("#feature maps of F and X should match: size(F,4)==size(X,4).");

  if (F.getSizeAtDim(4) != B.getSizeAtDim(1)) 
    throw conv3d_ex("#Bias should match the output feature map: size(F,5)==size(B,2).");

  // TODO: check the device type

  // size Y: the right size taking pad and stride into account
  if (pad[0]+pad[1]+X.getSizeAtDim(0) < F.getSizeAtDim(0) || 
    pad[2]+pad[3]+X.getSizeAtDim(1) < F.getSizeAtDim(1) ||
    pad[4]+pad[5]+X.getSizeAtDim(2) < F.getSizeAtDim(2) )
    throw conv3d_ex("Filter size should not be greater than feature map size.");

  mwSize szY[5];
  szY[0] = (pad[0]+X.getSizeAtDim(0)+pad[1] - F.getSizeAtDim(0))/stride[0] + 1;
  szY[1] = (pad[2]+X.getSizeAtDim(1)+pad[3] - F.getSizeAtDim(1))/stride[1] + 1;
  szY[2] = (pad[4]+X.getSizeAtDim(2)+pad[5] - F.getSizeAtDim(2))/stride[2] + 1;
  szY[3] = F.getSizeAtDim(4);
  szY[4] = X.getSizeAtDim(4);

  // create Y
  Y.setMxArray( createVol5d(szY, X.getDevice()) );

//#ifdef TM
//  tm.stop();
//  double te = tm.getElapsedTimeInMilliSec();
//  mexPrintf("conv3d::create_Y: %f\n", te);
//#endif // TM
}

void conv3d::check_X_size()
{
  // TODO: code refactoring. duplicate code with create_Y()

  // size Y: the right size taking pad and stride into account
  mwSize HY = (pad[0]+X.getSizeAtDim(0)+pad[1] - F.getSizeAtDim(0))/stride[0] + 1;
  mwSize WY = (pad[2]+X.getSizeAtDim(1)+pad[3] - F.getSizeAtDim(1))/stride[1] + 1;
  mwSize DY = (pad[4]+X.getSizeAtDim(2)+pad[5] - F.getSizeAtDim(2))/stride[2] + 1;
  mwSize MY = F.getSizeAtDim(4);
  mwSize NY = X.getSizeAtDim(4);

  if (HY != dY.getSizeAtDim(0) ||
    WY != dY.getSizeAtDim(1) || 
    DY != dY.getSizeAtDim(2) ||
    MY != dY.getSizeAtDim(3) ||
    NY != dY.getSizeAtDim(4) )
    throw conv3d_ex("In bprop(): size(dzdY) is inconsistent with X and F.");
}

void conv3d::create_dX()
{
  dX.setMxArray( createVol5dZerosLike(X) );
}

void conv3d::create_dF()
{
  dF.setMxArray( createVol5dZerosLike(F) );
}

void conv3d::create_dB()
{
  dB.setMxArray( createVol5dZerosLike(B) );
}


//// impl of cleanup after mex exit
void conv3d_releaseWhenUnloadMex()
{
#ifdef WITH_GPUARRAY
  release_cublas_context(); // used by _conv3d_blas_gpu
#endif // WITH_GPUARRAY
  sm_release();
}


//// impl of conv3d_ex
conv3d_ex::conv3d_ex(const char* msg)
  : runtime_error(msg)
{

}


//// Impl of factory_c3d_homebrew
conv3d* factory_c3d_homebrew::parse_and_create(int no, mxArray *vo[], int ni, mxArray const *vi[])
{
  // fprop or bprop?
  conv3d holder;
  int n_opt = -1;
  if (no == 1) {
    if ( ni < 3)
      throw conv3d_ex("Too few input arguments for fprop(). At least three: X, F, B.");

    holder.X.setMxArray( (mxArray*) vi[0] );
    holder.F.setMxArray( (mxArray*) vi[1] );
    holder.B.setMxArray( (mxArray*) vi[2] );
    if ( holder.X.getElemType() != mxSINGLE_CLASS || 
         holder.F.getElemType() != mxSINGLE_CLASS ||
         holder.B.getElemType() != mxSINGLE_CLASS) 
      throw conv3d_ex("The first three arguments X, F, B should be all SINGLE type,"
                      "be all gpuArray or be all mxArray.\n");

    holder.ct = conv3d::FPROP;
    n_opt = 3;
  } 
  else if (no == 3) {
    if ( ni < 4 ) 
      throw conv3d_ex("Too few input arguments for bprop(). At least four: X, F, B, dZdY.");

    holder.X.setMxArray( (mxArray*) vi[0] );
    holder.F.setMxArray( (mxArray*) vi[1] );
    holder.B.setMxArray( (mxArray*) vi[2] );
    holder.dY.setMxArray( (mxArray*) vi[3] );
    if (holder.X.getElemType() != mxSINGLE_CLASS || 
        holder.F.getElemType() != mxSINGLE_CLASS ||
        holder.B.getElemType() != mxSINGLE_CLASS ||
        holder.dY.getElemType() != mxSINGLE_CLASS)
      throw conv3d_ex("The first four arguments X, F, B, dZdY should be SINGLE type,"
                      "be all gpuArray or be all mxArray.\n");

    holder.ct = conv3d::BPROP;
    n_opt = 4;
  } 
  else {
    throw conv3d_ex("Unrecognized way of calling."
      "The output should be either Y (fprop) or [dX,dF,dB] (bprop). \n");
  }

  check_type(holder);

  set_options(holder, n_opt, ni, vi);

  bool is_fc = is_fullconnection(holder);

#ifdef WITH_GPUARRAY
  if ( xpuMxArrayTW::GPU == holder.X.getDevice() ) {
    if (is_fc) return new conv3d_blas_gpu_fc(holder);
    // default:
    return new conv3d_blas_gpu(holder);
  }
    
#endif // WITH_GPUARRAY

  return new conv3d_blas_cpu(holder);
}

void factory_c3d_homebrew::set_options(conv3d &holder, int opt_beg, int ni, mxArray const *vi[])
{
  // parse option/value pairs
  if ( ((ni-opt_beg)%2) != 0 )
    throw conv3d_ex("Imbalanced option/value pairs.");
  for (int i = opt_beg; i < ni; i+=2) {
    if (isStrEqual(vi[i], "stride"))   set_stride(holder, vi[i+1]);
    else if (isStrEqual(vi[i], "pad")) set_pad(holder, vi[i+1]);
    else                               throw conv3d_ex("Unrecognized option/value pairs.");
  } // for i
}

void factory_c3d_homebrew::set_stride(conv3d &holder, mxArray const *pa )
{
  if ( !setCArray<mwSize, 3>(pa, holder.stride) )
    throw conv3d_ex("The length of option stride should be either 1 or 3.");
}

void factory_c3d_homebrew::set_pad(conv3d &holder, mxArray const *pa )
{
  if ( !setCArray<mwSize, 6>(pa, holder.pad) )
    throw conv3d_ex("The length of option pad should be either 1 or 6.");
}

void factory_c3d_homebrew::check_type(const conv3d &holder)
{
  // X, F, B have been set
  xpuMxArrayTW::DEV_TYPE dt = holder.X.getDevice();
  bool flag = true;
  flag &= (dt == holder.F.getDevice());
  flag &= (dt == holder.B.getDevice());

  if (!flag) 
    throw conv3d_ex("In fprop(), X, F, B must be all gpuArray or mxArray.\n");

  // dY has been set?
  if ( holder.dY.pa_cpu != 0 ) 
    flag &= (dt == holder.dY.getDevice());

  if (!flag) 
    throw conv3d_ex("In bprop(), X, F, B, dZdY must be all gpuArray or mxArray.\n");
}

bool factory_c3d_homebrew::is_fullconnection(const conv3d &holder)
{
  return ( holder.X.getSizeAtDim(0) == holder.F.getSizeAtDim(0)  &&
           holder.X.getSizeAtDim(1) == holder.F.getSizeAtDim(1)  && 
           holder.X.getSizeAtDim(2) == holder.F.getSizeAtDim(2)  &&
           holder.pad[0]==0 && holder.pad[1]==0 &&
           holder.pad[2]==0 && holder.pad[3]==0 &&
           holder.pad[4]==0 && holder.pad[5]==0 );
}