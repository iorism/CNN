#pragma once
#include "conv3d.h"
#include "wrapperBlas.h"

//// conv3d: cpu version
struct conv3d_blas_cpu : public conv3d {
  conv3d_blas_cpu ();
  conv3d_blas_cpu (const conv3d& rhs);

  void fprop ();
  void bprop ();

private:
  // helper: fprop
  matw make_F_ ();
  matw make_Y_ (mwSize i);
  matw make_B_ ();

  // helper: bprop
  matw make_dY_ (mwSize i);
  matw make_dF_ ();
  matw make_dB_ ();

  // helper: the stacked matrix storing phiX or dphiX
  void init_convmat ();
  void free_convmat ();
  void vol_to_convmat   (xpuMxArrayTW &pvol, mwSize iInst); // im2row
  void vol_from_convmat (xpuMxArrayTW &pvol, mwSize iInst); // row2im
  matw convmat;

  void init_u ();
  void free_u ();
  matw u;

private:
  // helper for vol_to_convmat and convmat_to_vol
  enum DIR {VOL_TO_CONVMAT, VOL_FROM_CONVMAT};
  template<DIR how> void cpy_convmat_vol (xpuMxArrayTW &pvol, mwSize iInst);

};

//// helper 