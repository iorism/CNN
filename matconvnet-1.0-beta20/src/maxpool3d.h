#pragma once
#include "wrapperMx.h"
#include <stdexcept>

//// the transformer
struct maxpool3d {
  enum CALL_TYPE {FPROP, BPROP}; // way of calling 

  maxpool3d ();
  virtual ~maxpool3d ();

  // options
  CALL_TYPE ct;
  mwSize pad[6];
  mwSize pool[3];
  mwSize stride[3];
  // intermediate data: max elements index
  xpuMxArrayTW ind;
  // data at input/output port
  xpuMxArrayTW X, dX;
  xpuMxArrayTW Y, dY;

  // forward/backward propagation
  virtual void fprop () {};
  virtual void bprop () {};

  // helper: command
  static const char * THE_CMD;

protected:
  void check_dY_ind   ();
  void create_Y   ();
  void create_ind ();
  void create_dX  ();
};


//// exception: error message carrier
struct mp3d_ex : public std::runtime_error {
  mp3d_ex (const char *msg);
};

//// factory: select implementation
struct factory_mp3d {
  virtual maxpool3d* parse_and_create (int no, mxArray *vo[], int ni, mxArray const *vi[]) = 0;

};

struct factory_mp3d_homebrew : public factory_mp3d {
  virtual maxpool3d* parse_and_create (int no, mxArray *vo[], int ni, mxArray const *vi[]);

protected:
  void set_options(maxpool3d &h, int n_opt, int ni, mxArray const *vi[]);
  void set_pool   (maxpool3d &h, mxArray const *pa);
  void set_stride (maxpool3d &h, mxArray const *pa);
  void set_pad    (maxpool3d &h, mxArray const *pa);

  void check_padpool (const maxpool3d &h);
};

struct factory_mp3d_withcudnn : public factory_mp3d { 
  // 3D data not implemented in cudnn yet...could be the case in the future?
  virtual maxpool3d* parse_and_create (int no, mxArray *vo[], int ni, mxArray const *vi[]);
};
