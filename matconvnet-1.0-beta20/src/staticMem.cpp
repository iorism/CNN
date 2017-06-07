#pragma once

#include <cassert>
#include "staticMem.h"
#include "_staticMem.h"


//// Impl of buf_t
buf_t::buf_t()
: beg(0), nelem(0)
{

}

bool buf_t::is_need_realloc( size_t _nelem)
{
  return ( beg==0 || _nelem>nelem );
}


//// Impl of the public interface
float* sm_zeros (size_t nelem, xpuMxArrayTW::DEV_TYPE dt)
{
#ifdef WITH_GPUARRAY
  if (dt == xpuMxArrayTW::GPU) 
    return sm_zeros_gpu(nelem);
#endif // WITH_GPUARRAY
  
  assert(dt == xpuMxArrayTW::CPU);
  return sm_zeros_cpu(nelem);
}

float* sm_ones (size_t nelem, xpuMxArrayTW::DEV_TYPE dt)
{
#ifdef WITH_GPUARRAY
  if (dt == xpuMxArrayTW::GPU)
    return sm_ones_gpu(nelem);
#endif // WITH_GPUARRAY

  assert(dt == xpuMxArrayTW::CPU);
  return sm_ones_cpu(nelem);
}

void sm_release ()
{
#ifdef WITH_GPUARRAY
  sm_release_gpu();
#endif // WITH_GPUARRAY

  sm_release_cpu();
}