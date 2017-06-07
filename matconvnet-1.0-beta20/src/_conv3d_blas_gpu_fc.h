#pragma once
#include "_conv3d_blas_gpu.h"
#include "wrapperBlas.h"

//// conv3d: gpu version, full connection
struct conv3d_blas_gpu_fc : public conv3d_blas_gpu {

  conv3d_blas_gpu_fc (const conv3d& rhs) : conv3d_blas_gpu(rhs) {};

  virtual void fprop ();
  virtual void bprop ();

protected:
  matw make_XX_  (); // [HWDP, N]
  matw make_dXX_ (); // [HWDP, N]
  matw make_BB_  (); // [Q, 1]
  matw make_dBB_ (); // [Q, 1]
  matw make_YY_  (); // [Q, N]
  matw make_dYY_ (); // [Q, N]

protected: // helper for unit vector uu
  void initStaticMem_uu ();
  matw uu; // [1, N]
};
