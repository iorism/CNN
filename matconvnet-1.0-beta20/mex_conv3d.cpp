#include "mex.h"
#include "src/conv3d.h"
#include "src/wrapperMx.h"

namespace {

  void cleanup () {
    conv3d_releaseWhenUnloadMex();
  }
}

// "Y = MEX_CONV3D(X,F,B); forward pass"
// "[dX,dF,dB] = MEX_CONV3D(X,F,B, dY); backward pass"
// "MEX_CONV3D(..., 'stride',s, 'pad',pad); options"
void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
  mexAtExit( cleanup );

#ifdef WITH_CUDNN
  factory_c3d_withcudnn factory;
#else
  factory_c3d_homebrew factory;
#endif

  conv3d* h = 0; // TODO: consider unique_ptr?
  try {
    h = factory.parse_and_create(no,vo,ni,vi);
    assert(h != 0);

    // do the job and set output
    if (h->ct == conv3d::FPROP) {
      h->fprop();
      vo[0] = h->Y.getMxArray();
    }
    if (h->ct == conv3d::BPROP) {
      h->bprop();
      vo[0] = h->dX.getMxArray();
      vo[1] = h->dF.getMxArray();
      vo[2] = h->dB.getMxArray();
    }
    // done: cleanup
    safe_delete(h);

  } 
  catch (const conv3d_ex& e) {
    safe_delete(h);
    mexErrMsgTxt( e.what() );
  }

}