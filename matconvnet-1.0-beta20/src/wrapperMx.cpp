#include "wrapperMx.h"
#include "logmsg.h"


//// Impl of xpuMxArray
xpuMxArrayTW::xpuMxArrayTW()
{
  LOGMSG("xpuMxArrayTW Constructor\n");
  pa_cpu = 0;
  pa_gpu = 0;
  dt = CPU;
}

xpuMxArrayTW::~xpuMxArrayTW()
{
  LOGMSG("xpuMxArrayTW Destructor ");
  LOGMSG("[%d %d %d %d %d]",
    getSizeAtDim(0),getSizeAtDim(1),getSizeAtDim(2),
    getSizeAtDim(3),getSizeAtDim(4));

  pa_cpu = 0;
#ifdef WITH_GPUARRAY
  if (dt == GPU) {// always do this according to Matlab Doc
    assert(pa_gpu != 0);
    LOGMSG("(mxGPU hearder destroyed) ");
    mxGPUDestroyGPUArray(pa_gpu);
    pa_gpu = 0;
  }
#endif // WITH_GPUARRAY
  LOGMSG("\n");
}

xpuMxArrayTW::xpuMxArrayTW(const xpuMxArrayTW& rhs)
{
  LOGMSG("xpuMxArrayTW Copy-Constructor ");

  // always do these stuff
  dt     = rhs.dt;
  pa_cpu = rhs.pa_cpu;
  pa_gpu = 0;

#ifdef WITH_GPUARRAY
  if ( rhs.dt == xpuMxArrayTW::GPU ) { // hold its own
    LOGMSG("(GPU data header created)");
    pa_gpu = (mxGPUArray*)mxGPUCreateFromMxArray(pa_cpu);
  }
#endif // WITH_GPUARRAY

  LOGMSG("\n");
  return;
}

xpuMxArrayTW& xpuMxArrayTW::operator=(const xpuMxArrayTW& rhs)
{
  LOGMSG("xpuMxArrayTW Operator= ");

  // always do these stuff
  dt     = rhs.dt;
  pa_cpu = rhs.pa_cpu;
  pa_gpu = 0;

#ifdef WITH_GPUARRAY
  if ( rhs.dt == xpuMxArrayTW::GPU ) { // hold its own
    LOGMSG("(GPU data header created)");
    pa_gpu = (mxGPUArray*)mxGPUCreateFromMxArray(pa_cpu);
  }
#endif // WITH_GPUARRAY

  LOGMSG("\n");
  return *this;
}

mwSize xpuMxArrayTW::getNDims() const
{
  if (pa_cpu == 0) return 0;

#ifdef WITH_GPUARRAY
  if (dt == GPU) // always do this according to Matlab Doc
    return mxGPUGetNumberOfDimensions(pa_gpu);
#endif // WITH_GPUARRAY

  // else (dt == CPU)
  return mxGetNumberOfDimensions(pa_cpu);
}

mwSize xpuMxArrayTW::getSizeAtDim(mwSize dim) const
{
  if (pa_cpu == 0) return 0;

  mwSize ndim = getNDims();
  if (dim >= ndim) return 1;

#ifdef WITH_GPUARRAY
  if (dt == GPU)
    return (mxGPUGetDimensions(pa_gpu))[dim];
#endif // WITH_GPUARRAY

  // else (dt == CPU)
  return (mxGetDimensions(pa_cpu))[dim];
}

xpuMxArrayTW::DEV_TYPE xpuMxArrayTW::getDevice() const
{
  return dt;
}

mxClassID xpuMxArrayTW::getElemType() const
{
  if (dt == CPU)
    return mxGetClassID(pa_cpu);

#ifdef WITH_GPUARRAY
  return mxGPUGetClassID(pa_gpu);
#endif // WITH_GPUARRAY
}

void* xpuMxArrayTW::getDataBeg() const
{
  if (dt == CPU)
    return mxGetData(pa_cpu);

#ifdef WITH_GPUARRAY
  return mxGPUGetData(pa_gpu);
#endif // WITH_GPUARRAY
}

void xpuMxArrayTW::setMxArray(mxArray *pa)
{
  LOGMSG("xpuMxArrayTW::setMxArray() ");
  // forget the previous mxArray, attach to the new one
  pa_cpu = pa;
  dt = CPU;

#ifdef WITH_GPUARRAY
  if (mxIsGPUArray(pa)) {
    // release old gpuArray, if any
    if (pa_gpu != 0) {
      mxGPUDestroyGPUArray(pa_gpu);
      LOGMSG("(mxGPU hearder destroyed) ");
    }

    // point to the new one
    pa_gpu = (mxGPUArray*) mxGPUCreateFromMxArray(pa);
    dt = GPU;
  }
#endif // WITH_GPUARRAY
  LOGMSG("[%d %d %d %d %d]\n", getSizeAtDim(0),getSizeAtDim(1),getSizeAtDim(2),
    getSizeAtDim(3),getSizeAtDim(4));
}

mxArray* xpuMxArrayTW::getMxArray() const
{
  LOGMSG("xpuMxArrayTW::getMxArray() ");
  LOGMSG("[%d %d %d %d %d]\n", 
    getSizeAtDim(0),getSizeAtDim(1),getSizeAtDim(2),
    getSizeAtDim(3),getSizeAtDim(4));
  return pa_cpu;
}

//// Impl of shorthand
mxArray* createVol5d(mwSize sz[], xpuMxArrayTW::DEV_TYPE dt)
{
  if (dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(5, sz, mxSINGLE_CLASS, mxREAL);

#ifdef WITH_GPUARRAY
  mxGPUArray* tmp = mxGPUCreateGPUArray(5, sz, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  LOGMSG("createVol5d: on GPU, %d KB.\n", 
    toKB(sz[0]*sz[1]*sz[2]*sz[3]*sz[4], mxSINGLE_CLASS));
  
  mxArray* pa = mxGPUCreateMxArrayOnGPU(tmp);
  mxGPUDestroyGPUArray(tmp);
  return pa;
#endif // WITH_GPUARRAY
}

mxArray* createVol5dZeros(mwSize sz[], xpuMxArrayTW::DEV_TYPE dt)
{
  if (dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(5, sz, mxSINGLE_CLASS, mxREAL);

#ifdef WITH_GPUARRAY
  mxGPUArray* tmp = mxGPUCreateGPUArray(5, sz, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
  LOGMSG("createVol5dZeros: on GPU, %d KB.\n", 
    toKB(sz[0]*sz[1]*sz[2]*sz[3]*sz[4], mxSINGLE_CLASS) );

  mxArray* pa = mxGPUCreateMxArrayOnGPU(tmp);
  mxGPUDestroyGPUArray(tmp);
  return pa;
#endif // WITH_GPUARRAY
}

mxArray* createVol5dLike(const xpuMxArrayTW &rhs, mxClassID tp /*= mxSINGLE_CLASS*/)
{
  if (rhs.dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(mxGetNumberOfDimensions(rhs.pa_cpu), 
                                mxGetDimensions(rhs.pa_cpu), 
                                tp, mxREAL);

#ifdef WITH_GPUARRAY
  mxGPUArray* tmp = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(rhs.pa_gpu),
                                      mxGPUGetDimensions(rhs.pa_gpu),
                                      tp, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  LOGMSG("createVol5dLike: on GPU, %d KB.\n", 
    toKB(numel(rhs), tp) );

  mxArray* pa = mxGPUCreateMxArrayOnGPU(tmp);
  mxGPUDestroyGPUArray(tmp);
  return pa;
#endif // WITH_GPUARRAY
}

mxArray* createVol5dZerosLike(const xpuMxArrayTW &rhs, mxClassID tp /*= mxSINGLE_CLASS*/)
{
  if (rhs.dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(mxGetNumberOfDimensions(rhs.pa_cpu), 
    mxGetDimensions(rhs.pa_cpu), 
    tp, mxREAL); // 0s ensured

#ifdef WITH_GPUARRAY
  mxGPUArray* tmp = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(rhs.pa_gpu),
    mxGPUGetDimensions(rhs.pa_gpu),
    tp, mxREAL, MX_GPU_INITIALIZE_VALUES); // with 0s

  LOGMSG("createVol5dZerosLike: on GPU, %d KB.\n", 
    toKB(numel(rhs), tp));

  mxArray* pa = mxGPUCreateMxArrayOnGPU(tmp);
  mxGPUDestroyGPUArray(tmp);
  return pa;
#endif // WITH_GPUARRAY
}

mwSize numel(const xpuMxArrayTW &rhs)
{
#ifdef WITH_GPUARRAY
  if (rhs.dt == xpuMxArrayTW::GPU)
    return mxGPUGetNumberOfElements(rhs.pa_gpu);
#endif // WITH_GPUARRAY

  // else (rhs.dt == xpuMxArrayTW::CPU)
  return mxGetNumberOfElements(rhs.pa_cpu);
}
