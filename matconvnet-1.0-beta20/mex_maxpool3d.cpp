#include "mex.h"
#include "src/maxpool3d.h"
#include "src/wrapperMx.h"

// [Y,ind] = MEX_MAXPOOL3D(X); forward pass
// dZdX = MEX_MAXPOOL3D(dZdY, ind); backward pass
// MEX_MAXPOOL3D(..., 'pool',pool, 'stride',s, 'pad',pad); options
void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
#ifdef WITHCUDNN
  factory_mp3d_withcudnn factory;
#else
  factory_mp3d_homebrew factory;
#endif

  maxpool3d* h = 0; // TODO: consider unique_ptr?
  try {
    h = factory.parse_and_create(no, vo, ni, vi);
    assert( h != 0 );

    // do the job and set output
    if (h->ct == maxpool3d::FPROP) {
      h->fprop();
      vo[0] = h->Y.getMxArray();
      vo[1] = h->ind.getMxArray();
    }
    if (h->ct == maxpool3d::BPROP) {
      h->bprop();
      vo[0] = h->dX.getMxArray();
    }

    // done: cleanup
    safe_delete(h);
  }
  catch (const mp3d_ex e) {
    safe_delete(h);
    mexErrMsgTxt(e.what());
  }
}