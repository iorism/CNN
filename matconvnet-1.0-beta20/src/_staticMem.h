#pragma once
#include "wrapperMx.h"

//// buffer 
struct buf_t {
  buf_t ();
  
  virtual void realloc (size_t _nelem) = 0;
  virtual void dealloc () = 0;

  bool is_need_realloc (size_t _nelem);
  
  float* beg; // host or device raw data pointer
  int nelem; // #bytes = nelem * sizeof(et)
};

// CPU buffer
struct buf_cpu_t : public buf_t {
  void realloc (size_t _nelem);
  void dealloc ();
};

// GPU buffer
struct buf_gpu_t : public buf_t {
  void realloc (size_t _nelem);
  void dealloc ();
};


//// The CPU and GPU interface
float* sm_zeros_cpu (size_t nelem);
float* sm_ones_cpu (size_t nelem);
void   sm_release_cpu ();
//
float* sm_zeros_gpu (size_t nelem);
float* sm_ones_gpu (size_t nelem);
void   sm_release_gpu ();