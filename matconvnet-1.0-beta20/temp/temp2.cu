#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "../src/wrapperMx.h"



void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
  xpuMxArrayTW a;
  a.setMxArray( (mxArray*) vi[0]);
  mexPrintf("a.pa_gpu : %p\n", a.pa_gpu);

  xpuMxArrayTW b;
  b.setMxArray( (mxArray*) vi[0]);
  mexPrintf("b.pa_gpu : %p\n", b.pa_gpu);

  return;
}