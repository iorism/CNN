#include "_conv3d_cpu.h"

namespace {

// helper: sub volume attaching to big volume
struct subvol4D {
  float* beg;
  mwSize offset[4];
  mwSize size[4];
  mwSize stride[4];

  void copy_to_row           (matw& to,   mwSize row);
  void copy_and_inc_from_row (matw& from, mwSize row);
};

void subvol4D::copy_to_row (matw& the_mat, mwSize row)
{
  float* pe_mat = the_mat.beg + row; // ptr element matrix
  
  for (mwSize d3 = 0; d3 < size[3]; ++d3) {
    float* ptr_d3 = (offset[3] + d3)*stride[3] + this->beg;

    for (mwSize d2 = 0; d2 < size[2]; ++d2) {
      float* ptr_d2 = (offset[2] + d2)*stride[2] + ptr_d3;

      for (mwSize d1 = 0; d1 < size[1]; ++d1) {
        float* ptr_d1 = (offset[1] + d1)*stride[1] + ptr_d2;

        for (mwSize d0 = 0; d0 < size[0]; ++d0) {
          float* pe_vol = (offset[0] + d0)*stride[0] + ptr_d1; // ptr element volume
          
          // copy the single element
          *pe_mat = *pe_vol;
          
          // advance to next matrix element
          pe_mat += the_mat.H;
        } // d0
      } // d1
    } // d2
  } // d3
  
}

void subvol4D::copy_and_inc_from_row (matw& the_mat, mwSize row)
{
  float* pe_mat = the_mat.beg + row; // ptr element matrix

  for (mwSize d3 = 0; d3 < size[3]; ++d3) {
    float* ptr_d3 = (offset[3] + d3)*stride[3] + this->beg;

    for (mwSize d2 = 0; d2 < size[2]; ++d2) {
      float* ptr_d2 = (offset[2] + d2)*stride[2] + ptr_d3;

      for (mwSize d1 = 0; d1 < size[1]; ++d1) {
        float* ptr_d1 = (offset[1] + d1)*stride[1] + ptr_d2;

        for (mwSize d0 = 0; d0 < size[0]; ++d0) {
          float* pe_vol = (offset[0] + d0)*stride[0] + ptr_d1; // ptr element volume

          // copy and increment the single element
          *pe_vol += *pe_mat;

          // advance to next matrix element
          pe_mat += the_mat.H;
        } // d0
      } // d1
    } // d2
  } // d3
}


} // namespace


//// impl of public methods
conv3d_blas_cpu::conv3d_blas_cpu()
{

}

void conv3d_blas_cpu::fprop()
{
  create_Y();
  init_convmat();
  init_u(); 

  // iterate over each training instance
  mwSize N = getVolN(X);
  for (mwSize i = 0; i < N; i++) {
    // make phiX: the convolution matrix
    vol_to_convmat(X, i);

    // convolution: Y_ = phiX * F_
    matw F_ = make_F_();
    matw Y_ = make_Y_(i);
    AxBtoC(convmat, F_, Y_, true); // overwrite Y_

    // plus the bias: Y_ += u * B
    matw B_ = make_B_();
    AxBtoC(u, B_, Y_, false); // accumulation on Y_
  }

  free_u();
  free_convmat();
}

void conv3d_blas_cpu::bprop()
{
  create_dX();
  create_dF();
  create_dB();
  init_convmat();
  init_u();

  mwSize N = getVolN(X);
  matw dF_ = make_dF_();
  matw dB_ = make_dB_();
  for (mwSize i = 0; i < N; ++i) {
    // make phiX: the convolution matrix
    vol_to_convmat(X, i);

    // dF += phiX' * dY_
    matw dY_ = make_dY_(i);
    ATxBtoC(convmat, dY_, dF_, false); // accumulation on dF_

    // dB += u' * dY
    ATxBtoC(u, dY_, dB_, false); // accumulation on dB_

    // dphiX = dY * F'
    matw F_ = make_F_();
    // safe to reuse convmat memory, remember to overwrite it!
    AxBTtoC(dY_, F_, convmat, true);
    // dX(:,:,:,:,i) <-- dphiX
    vol_from_convmat(dX, i);
  }

  free_u();
  free_convmat();
}

conv3d::CALL_TYPE conv3d_blas_cpu::parse_and_set( int no, mxArray *vo[], int ni, mxArray const *vi[] )
{
  conv3d_blas_cpu::CALL_TYPE ct;
  int n_opt = -1;

  // fprop or bprop?
  if (no == 1) {
    if ( ni < 3 || !mxIsSingle(vi[0]) || !mxIsSingle(vi[1]) || !mxIsSingle(vi[2]) ) 
      mexErrMsgTxt(THE_CMD);

    ct = FPROP;
    n_opt = 3;
    X = (mxArray*) vi[0]; // we hereby guarantee that we won't change the inputs!
    F = (mxArray*) vi[1];
    B = (mxArray*) vi[2];
  } 
  else if (no == 3) {
    if ( ni < 4 ) 
      mexErrMsgTxt(THE_CMD);
    if ( !mxIsSingle(vi[0]) || !mxIsSingle(vi[1]) ||
         !mxIsSingle(vi[2]) || !mxIsSingle(vi[3]) )
      mexErrMsgTxt(THE_CMD);

    ct = BPROP;
    n_opt = 4;
    X  = (mxArray*) vi[0]; // we hereby guarantee that we won't change the inputs!
    F  = (mxArray*) vi[1];
    B  = (mxArray*) vi[2];
    dY = (mxArray*) vi[3];
  } 
  else {
    mexErrMsgTxt(THE_CMD);
  }

  // parse option/value pairs
  if ( ((ni-n_opt)%2) != 0 ) // imbalance option/value
    mexErrMsgTxt(THE_CMD);
  for (int i = n_opt; i < ni; i+=2) {
    if (isStrEqual(vi[i], "stride"))   this->set_stride(vi[i+1]);
    else if (isStrEqual(vi[i], "pad")) this->set_pad(vi[i+1]);
    else                               mexErrMsgTxt(THE_CMD);
  } // for i

  return ct;
}

//// impl of helpers
void conv3d_blas_cpu::set_stride( mxArray const *pa )
{
  mexErrMsgTxt("Option stride not implemented yet. Sorry..."
    "Currently the stride is always 1.\n");
  //if ( !setCArray<mwSize, 3>(pa, this->stride) )
  //  mexErrMsgTxt(THE_CMD);
}

void conv3d_blas_cpu::set_pad( mxArray const *pa )
{
  mexErrMsgTxt("Option pad not implemented yet. Sorry..."
    "Currently the pad is always 0.\n");
  //if ( !setCArray<mwSize, 6>(pa, this->pad) )
  //  mexErrMsgTxt(THE_CMD);
}

void conv3d_blas_cpu::create_Y()
{
  // check input X and filter F, B
  if (getVolP(F) != getVolM(X))  // #feature maps should match
    mexErrMsgTxt(THE_CMD);

  if (getVolQ(F) != mxGetNumberOfElements(B)) // #Bias should math the output
    mexErrMsgTxt(THE_CMD);

  if (getVolH(X) < getVolH(F) || // filter should not be greater than feature map
      getVolW(X) < getVolW(F) ||
      getVolD(X) < getVolD(F) )
    mexErrMsgTxt(THE_CMD);

  // size Y TODO: the right size taking pad and stride into account
  mwSize HY = getVolH(X) - getVolH(F) + 1;
  mwSize WY = getVolW(X) - getVolW(F) + 1;
  mwSize DY = getVolD(X) - getVolD(F) + 1;
  mwSize MY = getVolQ(F);
  mwSize NY = getVolN(X);

  // create Y
  Y = createVol5d(HY, WY, DY, MY, NY);
}

matw conv3d_blas_cpu::make_F_()
{
  matw F_;
  F_.beg = getDataBeg<float>(F);
  F_.H   = numelVol(F) * getVolP(F);
  F_.W   = getVolQ(F);

  return F_;
}

matw conv3d_blas_cpu::make_Y_(mwSize i)
{
  matw Y_;
  Y_.beg = getVolInstDataBeg<float>(Y, i);
  Y_.H   = numelVol(Y);
  Y_.W   = getSzAtDim<4>(Y);

  return Y_;
}

matw conv3d_blas_cpu::make_B_()
{
  matw B_;
  B_.beg = getDataBeg<float>(B);
  B_.H   = 1;
  B_.W   = mxGetNumberOfElements(B);

  return B_;
}

void conv3d_blas_cpu::create_dX()
{
  dX = createVol5dLike(X);
}

void conv3d_blas_cpu::create_dF()
{
  dF = createVol5dLike(F);
}

void conv3d_blas_cpu::create_dB()
{
  dB = createVol5dLike(B);
}

matw conv3d_blas_cpu::make_dX_(mwSize i)
{
  matw dX_;
  dX_.beg = getVolInstDataBeg<float>(dX, i);
  dX_.H   = numelVol(dX);
  dX_.W   = getSzAtDim<4>(dX);

  return dX_;
}

matw conv3d_blas_cpu::make_dY_(mwSize i)
{
  matw dY_;
  dY_.beg = getVolInstDataBeg<float>(dY, i);
  dY_.H   = numelVol(dY);
  dY_.W   = getSzAtDim<4>(dY);

  return dY_;
}

matw conv3d_blas_cpu::make_dF_()
{
  matw dF_;
  dF_.beg = getVolDataBeg<float>(dF);
  dF_.H   = numelVol(dF) * getSzAtDim<4>(dF);
  dF_.W   = getSzAtDim<5>(dF);

  return dF_;
}

matw conv3d_blas_cpu::make_dB_()
{
  matw dB_;
  dB_.beg = getVolDataBeg<float>(dB);
  dB_.H   = 1;
  dB_.W   = mxGetNumberOfElements(dB);
  
  return dB_;
}

void conv3d_blas_cpu::init_convmat()
{
  if (Y != 0) // in FPROP, Y has been set
    convmat.H = numelVol(Y);
  else if (dY != 0) // in BPROP, dY has been set
    convmat.H = numelVol(dY);
  else
    mexErrMsgTxt(THE_CMD);

  convmat.W = numelVol(F) * getVolP(F);
  mwSize nelem = convmat.H * convmat.W;
  convmat.beg = (float*)mxCalloc( nelem, sizeof(float) );
  // mxCalloc assures the initialization with all 0s ! 
}

void conv3d_blas_cpu::free_convmat()
{
  mxFree( (void*)convmat.beg );
}

void conv3d_blas_cpu::vol_to_convmat(const mxArray *pvol, mwSize iInst)
{
  // TODO: code refactoring. almost the same with vol_to_convmat

  // v: [H,   W,   D,   P]
  // F: [H',  W',  D',  P]
  // Y: [H'', W'', D'', 1]
  // convmat: [H''W''D''  H'W'D'P]

  // sub volume attaching to the big volume
  subvol4D sv;
  sv.beg = getVolInstDataBeg<float>(pvol, iInst);
  sv.size[0] = getVolH(F); 
  sv.size[1] = getVolW(F);
  sv.size[2] = getVolD(F);
  sv.size[3] = getVolP(F);
  sv.stride[0] = 1;
  sv.stride[1] = getVolH(pvol);
  sv.stride[2] = getVolH(pvol)*getVolW(pvol);
  sv.stride[3] = numelVol(pvol); // i.e., W*H*D

  // iterate over the big volume and set the offset for the sub volume
  mwSize row = 0;
  mwSize H = getVolH(pvol), W = getVolW(pvol), D = getVolD(pvol), P = getVolP(pvol);
  mwSize FH = getVolH(F), FW = getVolW(F), FD = getVolD(F); 

  sv.offset[3] = 0; // never slide at dim 4 !
  for (mwSize k = 0; k < (D - FD + 1); ++k) {
    sv.offset[2] = k;
    for (mwSize j = 0; j < (W - FW + 1); ++j) {
      sv.offset[1] = j;
      for (mwSize i = 0; i < (H - FH + 1); ++i) {
        sv.offset[0] = i;

        // copy to convmat(row, :)
        sv.copy_to_row(convmat, row);

        // step to next row, should be consistent with i,j,k,p
        ++row;
      } // i
    } // j
  }// k

}

void conv3d_blas_cpu::vol_from_convmat(mxArray *pvol, mwSize iInst)
{
  // TODO: code refactoring. almost the same with vol_to_convmat

  // v: [H,   W,   D,   P]
  // F: [H',  W',  D',  P]
  // Y: [H'', W'', D'', 1]
  // convmat: [H''W''D''  H'W'D'P]
  subvol4D sv;
  sv.beg = getVolInstDataBeg<float>(pvol, iInst);
  sv.size[0] = getVolH(F); 
  sv.size[1] = getVolW(F);
  sv.size[2] = getVolD(F);
  sv.size[3] = getVolP(F);
  sv.stride[0] = 1;
  sv.stride[1] = getVolH(pvol);
  sv.stride[2] = getVolH(pvol)*getVolW(pvol);
  sv.stride[3] = numelVol(pvol); // i.e., W*H*D

  // iterate over the big volume and set the offset for the sub volume
  mwSize row = 0;
  mwSize H = getVolH(pvol), W = getVolW(pvol), D = getVolD(pvol), P = getVolP(pvol);
  mwSize FH = getVolH(F), FW = getVolW(F), FD = getVolD(F); 

  sv.offset[3] = 0; // never slide at dim 4
  for (mwSize k = 0; k < (D - FD + 1); ++k) { // slide at dim 3
    sv.offset[2] = k;
    for (mwSize j = 0; j < (W - FW + 1); ++j) { // slide at dim 2
      sv.offset[1] = j;
      for (mwSize i = 0; i < (H - FH + 1); ++i) { // slide at dim 1
        sv.offset[0] = i;

        // copy to convmat(row, :)
        sv.copy_and_inc_from_row(convmat, row);

        // step to next row, should be consistent with i,j,k,p
        ++row;
      } // i
    } // j
  }// k

}

void conv3d_blas_cpu::init_u()
{
  if (Y != 0)
    u.H = numelVol(Y);
  else if (dY != 0)
    u.H = numelVol(dY);
  else
    mexErrMsgTxt(THE_CMD);

  u.W = 1;
  mwSize nelem = u.H * u.W ;
  u.beg = (float*)mxMalloc( nelem * sizeof(float) );

  // make sure all one
  for (int i = 0; i < nelem; i++)
    u.beg[i] = 1.0;
}

void conv3d_blas_cpu::free_u()
{
  mxFree( (void*)u.beg );
}