#include "cuda_runtime.h"
#include "_conv3d_blas_gpu_fc.h"
#include "_cu_helper.cuh"
#include "staticMem.h"

#include "logmsg.h"


void conv3d_blas_gpu_fc::fprop()
{
  create_Y();
  initStaticMem_uu(); 

  try {
    // YY_ = F_' * XX_
    matw XX_ = make_XX_();
    matw F_  = make_F_();
    matw YY_ = make_YY_();
    cu_ATxBtoC(F_, XX_, YY_, true);

    // YY_ += BB * uu
    matw BB_ = make_BB_();
    cu_AxBtoC(BB_,uu, YY_, false);

  } // try
  catch (const blas_ex& e) {
    throw conv3d_ex(e.what());
  }
  catch (const sm_ex& e) {
    throw conv3d_ex(e.what());
  }

}

void conv3d_blas_gpu_fc::bprop()
{
  check_X_size();
  create_dX();
  create_dF();
  create_dB();

  initStaticMem_uu();

  try {
    // dXX_ = F_ * dYY_
    matw F_   = make_F_();
    matw dYY_ = make_dYY_();
    matw dXX_ = make_dXX_();
    cu_AxBtoC(F_, dYY_, dXX_, true);

    // dF_ = XX_ * dYY_'
    matw XX_ = make_XX_();
    matw dF_ = make_dF_();
    cu_AxBTtoC(XX_,dYY_, dF_, true);

    // dBB_ = uu * dYY_'
    matw dBB_ = make_dBB_();
    cu_AxBTtoC(uu, dYY_, dBB_, true);

  }
  catch (const blas_ex& e) {
    throw conv3d_ex(e.what());
  }


}


//// Impl of helpers
matw conv3d_blas_gpu_fc::make_XX_()
{
  matw XX;
  XX.beg = (float*)X.getDataBeg();
  XX.H   = numelVol(X) * X.getSizeAtDim(3);
  XX.W   = X.getSizeAtDim(4);

  return XX;
}

matw conv3d_blas_gpu_fc::make_dXX_()
{
  matw dXX;
  dXX.beg = (float*)dX.getDataBeg();
  dXX.H   = numelVol(dX) * dX.getSizeAtDim(3);
  dXX.W   = X.getSizeAtDim(4);

  return dXX;
}

matw conv3d_blas_gpu_fc::make_BB_()
{
  matw BB;
  BB.beg = (float*)B.getDataBeg();
  BB.H   = F.getSizeAtDim(4);
  BB.W   = 1;

  return BB;
}

matw conv3d_blas_gpu_fc::make_dBB_()
{
  matw dBB;
  dBB.beg = (float*)dB.getDataBeg();
  dBB.H   = F.getSizeAtDim(4);
  dBB.W   = 1;

  return dBB;
}

matw conv3d_blas_gpu_fc::make_YY_()
{
  matw YY;
  YY.beg = (float*)Y.getDataBeg();
  YY.H   = Y.getSizeAtDim(3);
  YY.W   = Y.getSizeAtDim(4);

  return YY;
}

matw conv3d_blas_gpu_fc::make_dYY_()
{
  matw dYY;
  dYY.beg = (float*)dY.getDataBeg();
  dYY.H   = dY.getSizeAtDim(3);
  dYY.W   = dY.getSizeAtDim(4);

  return dYY;
}

//// Imple of helpers for unit vector uu
void conv3d_blas_gpu_fc::initStaticMem_uu()
{
  uu.H = 1;
  uu.W = X.getSizeAtDim(4);

  mwSize nelem = uu.W ;

  uu.beg = sm_ones(nelem, X.getDevice());
}
