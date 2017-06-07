#pragma once

#include "mex.h"

#ifdef WITH_GPUARRAY
#include "gpu/mxGPUArray.h"
#else
struct mxGPUArray;
#endif // WITH_GPUARRAY

#include <cstring>
#include <numeric>
#include <cassert>


//// shorthand for mxArray
inline bool isStrEqual (mxArray const *pa, char const * str) {
  char* tmp = mxArrayToString(pa);
  bool isEqual = (0 == strcmp(tmp, str)) ? true : false;
  mxFree(tmp);
  return isEqual;
}

template<typename T> inline 
T* getDataBeg(mxArray const *pa) {
  return ( (T*)mxGetData(pa) );
}

template<typename T, int N> inline 
bool setCArray (mxArray const *pa, T arr[]) {
  if (!mxIsDouble(pa)) mexErrMsgTxt("setCArray: pa must be double matrix\n");

  bool flag_success =  true;
  mwSize nelem = mxGetNumberOfElements(pa);
  if (nelem == N) {
    for (int i = 0; i < N; ++i)
      arr[i] = T(*(getDataBeg<double>(pa) + i));
  } else if (nelem == 1) {
    for (int i = 0; i < N; ++i)
      arr[i] = T(mxGetScalar(pa));
  } else {
    flag_success = false;
  }
  return flag_success;
}

inline size_t sizeofMxType (mxClassID t) {
  if (t == mxSINGLE_CLASS) return 4;
  if (t == mxINT32_CLASS)  return 4;
  if (t == mxDOUBLE_CLASS) return 8;

  return 0; // TODO: more types if necessary
}

//// Thin wrapper for mxArray, which can be mxGPUArray. Never owns data.
struct xpuMxArrayTW {
  enum DEV_TYPE {CPU, GPU};

  xpuMxArrayTW  ();
  ~xpuMxArrayTW ();
  
  xpuMxArrayTW            (const xpuMxArrayTW& rhs);
  xpuMxArrayTW& operator= (const xpuMxArrayTW& rhs);

  void     setMxArray (mxArray *pa); // never owns the data
  mxArray* getMxArray () const;

  mwSize    getNDims     () const;
  mwSize    getSizeAtDim (mwSize dim) const;
  DEV_TYPE  getDevice    () const;
  mxClassID getElemType  () const;
  void*     getDataBeg   () const;

// private:
  mxArray*    pa_cpu;
  mxGPUArray* pa_gpu;
  DEV_TYPE    dt;

};


//// Shorthand for xpuMxArrayTW (as Volume 3D + 2D): creation
mxArray* createVol5d (mwSize sz[], xpuMxArrayTW::DEV_TYPE dt);

mxArray* createVol5dZeros (mwSize sz[], xpuMxArrayTW::DEV_TYPE dt);

mxArray* createVol5dLike (const xpuMxArrayTW &rhs, mxClassID tp = mxSINGLE_CLASS);

mxArray* createVol5dZerosLike(const xpuMxArrayTW &rhs, mxClassID tp = mxSINGLE_CLASS);


//// Shorthand for xpuMxArrayTW (as Volume 3D + 2D): data access
template<typename T> inline 
T* getVolDataBeg(const xpuMxArrayTW &rhs, mwSize iVol = 0) {
  mwSize stride = rhs.getSizeAtDim(0) * rhs.getSizeAtDim(1) * rhs.getSizeAtDim(2);
  T* pbeg = (T*)rhs.getDataBeg();
  return (pbeg + iVol*stride);
}

template<typename T> inline 
T* getVolInstDataBeg(const xpuMxArrayTW &rhs, mwSize iInst = 0) {
  mwSize stride = rhs.getSizeAtDim(0) * rhs.getSizeAtDim(1) * 
                  rhs.getSizeAtDim(2) * rhs.getSizeAtDim(3);
  T* pbeg = (T*)rhs.getDataBeg();
  return (pbeg + iInst*stride);
}

inline mwSize numVol (const xpuMxArrayTW &rhs) {
  mwSize ndim = rhs.getNDims();
  if (ndim <= 3 ) return 1;

  mwSize n = 1;
  for (mwSize i = 3; i < ndim; ++i) n *= rhs.getSizeAtDim(i);
  return n;
}
 
inline mwSize numelVol (const xpuMxArrayTW &rhs) {
  return rhs.getSizeAtDim(0) *
         rhs.getSizeAtDim(1) *
         rhs.getSizeAtDim(2);
}


//// Shorthand for xpuMxArrayTW
mwSize numel (const xpuMxArrayTW &rhs);


//// Miscellaneous
template<typename T> inline 
void safe_delete (T* &ptr) { // "safe": delete if zero, set to zero after deletion
  if (ptr != 0) {
    delete ptr;
    ptr = 0;
  }
}
