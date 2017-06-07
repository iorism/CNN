#include "matrix.h"
#include "wrapperMx.h"
#include "maxpool3d.h"
#include "_maxpool3d_cpu.h"

#ifdef WITH_GPUARRAY
#include "gpu/mxGPUArray.h"
#include "_maxpool3d_gpu.h"
#endif // WITH_GPUARRAY

//// Impl of maxpool3d
const char* maxpool3d::THE_CMD = 
  "Bad input or output arguments. The right way to call:\n"
  "[Y,ind] = MEX_MAXPOOL3D(X); forward pass\n"
  "dZdX = MEX_MAXPOOL3D(dZdY, ind); backward pass\n"
  "MEX_MAXPOOL3D(..., 'pool',pool, 'stride',s, 'pad',pad); options\n"
  "X, dZdY must be single, ind must be double;\n";

maxpool3d::maxpool3d()
{
  pool[0] = pool[1] = pool[2] = 2;
  stride[0] = stride[1] = stride[2] = 2;
  pad[0] = pad[1] = pad[2] = pad[3] = pad[4] = pad[5] = 0;
}

maxpool3d::~maxpool3d()
{
}

void maxpool3d::check_dY_ind()
{
  if (dY.getDevice() != ind.getDevice())
    throw mp3d_ex("In bprop(): dY and ind must be both gpuArray or CPU mxArray.\n");
}

void maxpool3d::create_Y()
{

  if (pad[0]+pad[1]+X.getSizeAtDim(0) < pool[0] || 
    pad[2]+pad[3]+X.getSizeAtDim(1) < pool[1] ||
    pad[4]+pad[5]+X.getSizeAtDim(2) < pool[2] )
    throw mp3d_ex("pooling window size should not be greater than feature map size.");

  // size Y: the right size taking into account pad and stride
  mwSize szY[5];
  szY[0] = (pad[0]+X.getSizeAtDim(0)+pad[1] - pool[0])/stride[0] + 1;
  szY[1]= (pad[2]+X.getSizeAtDim(1)+pad[3] - pool[1])/stride[1] + 1;
  szY[2] = (pad[4]+X.getSizeAtDim(2)+pad[5] - pool[2])/stride[2] + 1;
  szY[3] = X.getSizeAtDim(3);
  szY[4] = X.getSizeAtDim(4);

  // create Y
  Y.setMxArray( createVol5d(szY, X.dt) );
}

void maxpool3d::create_ind()
{
  ind.setMxArray( createVol5dLike(Y, mxINT32_CLASS) );
}

void maxpool3d::create_dX()
{
  if (dX.getNDims() != 0) return; // already created

  // size dX: the right size taking into account pad and stride
  mwSize szdX[5] = {0,0,0,1,1};
  szdX[0] = stride[0]*(dY.getSizeAtDim(0)-1) - (pad[0]+pad[1]) + pool[0];
  szdX[1] = stride[1]*(dY.getSizeAtDim(1)-1) - (pad[2]+pad[3]) + pool[1];
  szdX[2] = stride[2]*(dY.getSizeAtDim(2)-1) - (pad[4]+pad[5]) + pool[2];
  szdX[3] = dY.getSizeAtDim(3);
  szdX[4] = dY.getSizeAtDim(4);

  // create dX
  dX.setMxArray( createVol5dZeros(szdX, dY.dt) );
}


//// Impl of mp3d_ex
mp3d_ex::mp3d_ex(const char* msg)
  : runtime_error(msg)
{

}


//// Impl of factory

maxpool3d* factory_mp3d_homebrew::parse_and_create(int no, mxArray *vo[], int ni, mxArray const *vi[])
{
  if (ni < 1) 
    throw mp3d_ex("Too few input arguments.");

  // fprop or bprop?
  maxpool3d holder;
  int opt_beg = -1;
  xpuMxArrayTW::DEV_TYPE dt;

  if (no == 2) { // fprop
    holder.X.setMxArray( (mxArray*) vi[0] ); // we won't change it
    dt = holder.X.getDevice();

    if ( ni < 1 || (holder.X.getElemType() != mxSINGLE_CLASS) ) 
      throw mp3d_ex("For fprop(), there should be at least one input, X, of SINGLE type,"
                    "be all gpuArray or be all mxArray.\n");

    holder.ct = maxpool3d::FPROP;
    opt_beg = 1;
  } 
  else if (no == 1) { // bprop
    holder.dY.setMxArray( (mxArray*)  vi[0]);
    holder.ind.setMxArray( (mxArray*) vi[1]);
    dt = holder.dY.getDevice();

    if ( ni < 2 || 
         holder.dY.getElemType()  != mxSINGLE_CLASS || 
         holder.ind.getElemType() != mxINT32_CLASS) 
      throw mp3d_ex("For bprop(): there should be at least 3 arguments, dzdY, ind.\n"
        "The dzdY must be SINGLE, the max index ind must be int32,"
        "they should be both gpuArray or be both mxArray.\n");

    holder.ct = maxpool3d::BPROP;
    opt_beg = 2;
  } 
  else {
    throw mp3d_ex("Unrecognized arguments/way of calling. "
      "The output should be either [Y, ind] (fprop) or ind (bprop). ");
  }

  // if bprop: create dX if szX is provided args:(dzdY, ind, szX)
  if (holder.ct == maxpool3d::BPROP) {
    if (ni >= 3 && !mxIsChar(vi[2]) ) {
      // szX provided, check it
      if (!mxIsDouble(vi[2])) mexErrMsgTxt("setCArray: pa must be double matrix\n");
      double *ptr = (double*)mxGetData(vi[2]);
      mwSize nelem = mxGetNumberOfElements(vi[2]);
      if (nelem > 5 || nelem < 3) 
        throw mp3d_ex("The third argument must be: 3 <= numel(szX) <= 5.\n");

      // get szX
      mwSize szX[5];
      szX[3] = szX[4] = 1;
      for (int i = 0; i < nelem; ++i) szX[i] = (mwSize) ptr[i];
      
      // create the dX
      holder.dX.setMxArray( createVol5dZeros(szX, holder.dY.dt) );

      // reset the option beginning
      opt_beg = 3;
    } 
    else // issue the warning
      mexWarnMsgTxt("For bprop(), the calling method with 2 args:\n"
        "[...] = mex_maxpool3d(dzdY, ind, ...)\n"
        "is deprecated, because this could cause ambiguity when inferring input X size.\n"
        "Use the new one to specify the size for input X (or dzdX) explicitly:\n"
        "[...] = mex_maxpool3d(dzdY, ind, szX,...)\n");
  }

  // set options
  set_options(holder, opt_beg, ni, vi);

  // check validity
  check_padpool(holder);

  // create the desired worker and set the parameters
#ifdef WITH_GPUARRAY
  if (dt == xpuMxArrayTW::GPU)
    return new maxpool3d_gpu(holder);
  else
    return new maxpool3d_cpu(holder);
#else
  return new maxpool3d_cpu(holder);
#endif // WITH_GPUARRAY
}

void factory_mp3d_homebrew::set_options(maxpool3d &holder, int opt_beg, int ni, mxArray const *vi[])
{
  // parse option/value pairs
  if ( ((ni-opt_beg)%2) != 0 )
    throw mp3d_ex("Imbalanced option/value pairs.");
  for (int i = opt_beg; i < ni; i+=2) {
    if      (isStrEqual(vi[i], "pool"))   set_pool(holder, vi[i+1]);
    else if (isStrEqual(vi[i], "stride")) set_stride(holder, vi[i+1]);
    else if (isStrEqual(vi[i], "pad"))    set_pad(holder, vi[i+1]);
    else                                  throw mp3d_ex("Unrecognized option/value pairs.");
  } // for i
}

void factory_mp3d_homebrew::set_pool(maxpool3d &h, mxArray const *pa )
{
  if ( !setCArray<mwSize, 3>(pa, h.pool) )
    throw mp3d_ex("The length of option pool must be 1 or 3.");
}

void factory_mp3d_homebrew::set_stride(maxpool3d &h, mxArray const *pa )
{
  if ( !setCArray<mwSize, 3>(pa, h.stride) )
    throw mp3d_ex("The length of option stride must be 1 or 3.");
}

void factory_mp3d_homebrew::set_pad(maxpool3d &h, mxArray const *pa )
{
  if ( !setCArray<mwSize, 6>(pa, h.pad) )
    throw mp3d_ex("The length of option pad must be 1 or 6.");
}

void factory_mp3d_homebrew::check_padpool(const maxpool3d &h)
{
  if ( (h.pad[0] + h.pad[1]) >= h.pool[0] ||
       (h.pad[2] + h.pad[3]) >= h.pool[1] ||
       (h.pad[4] + h.pad[5]) >= h.pool[2] )
    throw mp3d_ex("Pool size must be strictly larger than (sum of lower, higher) pad size.");
}

