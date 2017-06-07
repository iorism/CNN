#include <cublas_v2.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "../src/wrapperMx.h"
#include "../src/wrapperBlas.h"
#include "../src/Timer.h"


matw makeit (xpuMxArrayTW& rhs) {
  matw A;
  A.beg = (float*) rhs.getDataBeg();
  A.H = rhs.getSizeAtDim(0);
  A.W = rhs.getSizeAtDim(1);

  return A;
}

matw create_Y_ (mwSize M, mwSize N) {
  mwSize sz[2];
  sz[0] = M; sz[1] = N;
  mxGPUArray* tmp = mxGPUCreateGPUArray(2, sz, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

  matw Y;
  Y.beg = (float*)mxGPUGetData(tmp);
  Y.H = M;
  Y.W = N;

  mxGPUDestroyGPUArray(tmp);
  return Y;
}


void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
  // input
  xpuMxArrayTW a;
  a.setMxArray( (mxArray*) vi[0]);
  matw convmat = makeit(a);

  xpuMxArrayTW b;
  b.setMxArray( (mxArray*) vi[1]);
  matw F_ = makeit(b);

  // output
  matw Y_ = create_Y_(convmat.H, F_.W);

  // do it
  mexPrintf("[%d %d] [%d %d]\n", convmat.H, convmat.W, F_.H, F_.W);

  const int N = 1500;
  mexPrintf("num inst = %d\n", N);

  cublasHandle_t hd;
  cublasCreate(&hd);

  Timer tm;
  double time = 0.0;

  for (int i = 0; i < N; ++i) {
    float alpha = 1.0;
    float beta = 0.0;

    tm.start();

    cublasStatus_t st = cublasSgemm(
      hd,
      CUBLAS_OP_N, CUBLAS_OP_N,
      (int)convmat.H, (int)F_.W, (int)convmat.W,
      &alpha,
      (float*)convmat.beg, (int)convmat.H,
      (float*)F_.beg, (int)F_.H,
      &beta,
      (float*)Y_.beg, (int)Y_.H);

    tm.stop();
    time += tm.getElapsedTimeInMilliSec();
    
  }

  mexPrintf("time = %f\n", time);
  
  return;
}