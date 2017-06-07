#pragma once

inline size_t ceil_divide (size_t a, size_t b) {
  return (a + b - 1)/b;
}


#if __CUDA_ARCH__ >= 200
const int CU_NUM_THREADS = 1024; 
#else
const int CU_NUM_THREADS = 512; 
#endif

//// helper: setting initial value
template<typename T>
__global__ void kernelSetZero (T* beg, size_t len) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < len) beg[ind] = static_cast<T>(0);
}

template<typename T>
__global__ void kernelSetOne (T* beg, size_t len) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < len) beg[ind] = static_cast<T>(1);
}