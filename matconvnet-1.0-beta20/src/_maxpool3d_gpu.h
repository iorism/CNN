#pragma once
#include "maxpool3d.h"

struct maxpool3d_gpu : public maxpool3d {
  maxpool3d_gpu ();
  maxpool3d_gpu (const maxpool3d &obj);

  void fprop ();
  void bprop ();
};