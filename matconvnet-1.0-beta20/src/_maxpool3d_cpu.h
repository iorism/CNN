#pragma once
#include "maxpool3d.h"

struct maxpool3d_cpu : public maxpool3d {
  maxpool3d_cpu ();
  maxpool3d_cpu (const maxpool3d &obj);

  void fprop ();
  void bprop ();

};