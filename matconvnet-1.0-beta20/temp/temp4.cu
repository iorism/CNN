#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "../src/wrapperMx.h"
#include "../src/wrapperBlas.h"
#include "../src/Timer.h"


void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
  conv3d_blas_gpu h;
  
  h.X.setMxArray( (mxArray*) vi[0] );

  h.init_convmat();


}