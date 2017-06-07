#pragma once
#include "wrapperMx.h"

#ifdef VB
#define LOGMSG mexPrintf
#else
#define LOGMSG(...)
#endif // VB

inline size_t toMB (size_t n, mxClassID t) {
  return (n * sizeofMxType(t) / 1e6);
}

inline size_t toKB (size_t n, mxClassID t) {
  return (n * sizeofMxType(t) / 1e3);
}