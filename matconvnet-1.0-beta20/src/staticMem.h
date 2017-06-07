#pragma once
#include <stdexcept>
#include "wrapperMx.h"

// static memory across mex calling, all zeros
float* sm_zeros (size_t nelem, xpuMxArrayTW::DEV_TYPE dt);

// static memory across mex calling, all ones
float* sm_ones (size_t nelem, xpuMxArrayTW::DEV_TYPE dt);

// release all (called when unloading mex file)
void sm_release ();

// exception: error message carrier
struct sm_ex : public std::runtime_error {
  sm_ex (const char* msg) : runtime_error(msg) {};
};