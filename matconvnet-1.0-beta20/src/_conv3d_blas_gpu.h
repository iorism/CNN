#pragma once
#include "conv3d.h"
#include "wrapperBlas.h"

//// conv3d: gpu version TODO: impl idiom
struct conv3d_blas_gpu : public conv3d {
  conv3d_blas_gpu ();
  conv3d_blas_gpu (const conv3d& rhs);

  virtual void fprop ();
  virtual void bprop ();

  // helper types for implementation
  struct vol {
    int sz[3];
    int szProd[3];

    void set_sz (const xpuMxArrayTW &rhs) {
      for (int i = 0; i < 3; ++i) sz[i] = rhs.getSizeAtDim(i);
      set_szProd();
    }

  private:
    void set_szProd () {
      szProd[0] = sz[0];
      szProd[1] = szProd[0] * sz[1];
      szProd[2] = szProd[1] * sz[2];
    }
  };

  struct CpyVolConvmatImpl {
    // Source, Target data
    float *vol4d_beg, *convmat_beg;
    // volume size for X(or dX), F, Y (or dY)
    vol X, F, Y;
    // #input feature map
    int P;
    // pre-computation: should always be Y.szProd[2]*P
    int YP;
    // other information
    int stride[3];
    int pad[6];
  };

protected:
  // helper: fprop
  matw make_F_ ();
  matw make_Y_ (mwSize i);
  matw make_B_ ();

  // helper: bprop
  matw make_dY_ (mwSize i);
  matw make_dF_ ();
  matw make_dB_ ();

protected: // helper: the stacked matrix storing phiX or dphiX
  CpyVolConvmatImpl make_initial_CpyVolConvmatImpl (const xpuMxArrayTW &vol);

  void initStaticMem_convmat ();
  void vol_to_convmat   (CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst); // im2row
  void vol_from_convmat (CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst); // row2im
  matw convmat;

protected: // helper for unit vector u
  void initStaticMem_u ();
  matw u;
};