#pragma once
#include "wrapperMx.h"
#include <stdexcept>


//// the transformer
struct conv3d {
  enum CALL_TYPE {FPROP, BPROP}; // way of calling

  conv3d ();
  virtual ~conv3d ();

  // options
  CALL_TYPE ct;
  mwSize pad[6];
  mwSize stride[3];
  // intermediate data: filter and bias
  xpuMxArrayTW F, dF;
  xpuMxArrayTW B, dB;
  // data at input/output port
  xpuMxArrayTW X, dX;
  xpuMxArrayTW Y, dY;

  // forward/backward propagation
  virtual void fprop () {};
  virtual void bprop () {};

  // helper: command
  static const char * THE_CMD;

protected:
  // helper for fprop
  void create_Y ();
  // helper for bprop
  void check_X_size ();
  void create_dX ();
  void create_dF ();
  void create_dB ();
};


//// release after mex unloaded (registered by mexAtExit)
void conv3d_releaseWhenUnloadMex ();

//// exception: error message carrier
struct conv3d_ex : public std::runtime_error {
  conv3d_ex (const char* msg);
};


//// factory: select implementation
struct factory_c3d {
  virtual conv3d* parse_and_create (int no, mxArray *vo[], int ni, mxArray const *vi[]) = 0;
};

struct factory_c3d_homebrew : public factory_c3d {
  virtual conv3d* parse_and_create (int no, mxArray *vo[], int ni, mxArray const *vi[]);

protected:
  void check_type        (const conv3d &holder);
  bool is_fullconnection (const conv3d &holder);

  void set_options (conv3d &holder, int n_opt, int ni, mxArray const *vi[]);
  void set_stride  (conv3d &holder, mxArray const *pa);
  void set_pad     (conv3d &holder, mxArray const *pa);
  
};

struct factory_c3d_withcudnn : public factory_c3d { 
  // 3D data not implemented in cudnn yet...could be the case in the future?
  virtual conv3d* parse_and_create (int no, mxArray *vo[], int ni, mxArray const *vi[]);
};
