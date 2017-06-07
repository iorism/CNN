#include "staticMem.h"
#include "_staticMem.h"
#ifndef WITH_GPUARRAY
#error WITH_GPUARRAY macro must be defined in order to compile this file.
#endif
#include <cuda_runtime.h>
#include "_cu_helper.cuh"


//// static buffers
static buf_gpu_t bufZeros_gpu;
static buf_gpu_t bufOnes_gpu;


//// Impl of buf_gpu_t
void buf_gpu_t::realloc( size_t _nelem )
{
  dealloc();
  
  void* tmp;
  cudaError_t st = cudaMalloc(&tmp, _nelem*sizeof(float));
  if ( cudaSuccess != st )
    throw sm_ex("staticMem: Out of GPU memory.\n");

  nelem = _nelem;
  beg = (float*)tmp;
}

void buf_gpu_t::dealloc()
{
  if (beg != 0) {
    if ( cudaSuccess != cudaFree((void*)beg) )
      throw sm_ex("staticMem: error when releasing GPU memory.\n");
  }
  beg = 0;
  nelem = 0;
}


//// Impl of GPU interface
float* sm_zeros_gpu (size_t nelem)
{
  if ( bufZeros_gpu.is_need_realloc(nelem) ) {
    bufZeros_gpu.realloc(nelem);

    // set initial value
    dim3 sz_blk( ceil_divide(nelem,CU_NUM_THREADS));
    dim3 sz_thd(CU_NUM_THREADS );
    kernelSetZero<float><<<sz_blk,sz_thd>>>(bufZeros_gpu.beg, nelem);
  }
  return bufZeros_gpu.beg;
}

float* sm_ones_gpu (size_t nelem)
{
  if ( bufOnes_gpu.is_need_realloc(nelem) ) {
    bufOnes_gpu.realloc(nelem);

    // set initial value
    dim3 sz_blk( ceil_divide(nelem,CU_NUM_THREADS));
    dim3 sz_thd(CU_NUM_THREADS );
    kernelSetOne<float><<<sz_blk,sz_thd>>>(bufOnes_gpu.beg, nelem);
  }
  return bufOnes_gpu.beg;
}

void   sm_release_gpu ()
{
  bufZeros_gpu.dealloc();
  bufOnes_gpu.dealloc();
}