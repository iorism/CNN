#pragma once
#include "mex.h"
#include <stdexcept>

// 2D matrix thin wrapper over raw data pointer (host or device)
// presume continuous memory
// caller owns data and ensures the initialization (the size)/validity (host or device pointer) 
struct matw {
  float *beg;
  mwSize H, W;
};

// blas error message carrier (mainly from cublas?)
struct blas_ex : public std::runtime_error {
  blas_ex (const char* msg);
};

// blas (cpu): no context needed, just use the functions
// cublas (gpu): context initialization is automatic, release must be manual
void release_cublas_context ();

// A*B + C -> C (accumulation) or A*B -> C(overwrite)
void AxBtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

// AT*B + C -> C (accumulation) or AT*B -> C (overwrite) 
void ATxBtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

// A*BT + C -> C (accumulation) or A*BT -> C (overwrite)
void AxBTtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

// A*B + C -> C (accumulation) or A*B -> C(overwrite)
void cu_AxBtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

// AT*B + C -> C (accumulation) or AT*B -> C (overwrite) 
void cu_ATxBtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

// A*BT + C -> C (accumulation) or A*BT -> C (overwrite)
void cu_AxBTtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

