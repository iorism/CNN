#include "mex.h"
#include "gpu/mxGPUArray.h"

//struct mmw {
//
//  // mmw () = delete;
//
//  void attach (mxArray const *pm);
//  void attach (void* pd, mwSize ndim, mwSize sz[], mxClassID et, );
//
//
//
//};

mwSize getndims (mxArray const *pm) {
  mwSize r;

  if (mxIsGPUArray(pm)) {
    mxGPUArray const * pmg = mxGPUCreateFromMxArray(pm);
    r = mxGPUGetNumberOfDimensions(pmg);
    mxGPUDestroyGPUArray(pmg);
  }
  else {
    r = mxGetNumberOfDimensions(pm);
  }
  return r;
}

mwSize getSzAtDim (mxArray const *pm) {
  mwSize r;

  if (mxIsGPUArray(pm)) {
    mxGPUArray const * pmg = mxGPUCreateFromMxArray(pm);
    r = mxGPUGetNumberOfDimensions(pmg);
    mxGPUDestroyGPUArray(pmg);
  }
  else {
    r = mxGetNumberOfDimensions(pm);
  }
  return r;

}


void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
  mxArray const * pa = vi[0];

  mwSize ndim = getndims(pa);
  //mexPrintf("ndim = %d\n", ndim);

  //mwSize ndim = mxGetNumberOfDimensions(pa);
  //mexPrintf("ndim = %d\n", ndim);

  //mwSize const * dim = mxGetDimensions(pa);
  //for (int i = 0; i < ndim; ++i)
  //  mexPrintf("dim %d = %d\n", i, dim[i]);


  return;
}