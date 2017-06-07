#include "wrapperBlas.h"

#ifdef WITH_GPUARRAY
#include <cublas_v2.h>


//// cublas monitor, a singleton Design Pattern
class cublasMon
{
public:
  static cublasMon& get_this () {
    static cublasMon instance; 
    return instance;
  }

  cublasHandle_t& get_cublasHandler () {
    if (!isInit) init_cublas();
    return hd;
  }

  void init_cublas () {
    if (!isInit) {
      if (CUBLAS_STATUS_SUCCESS != cublasCreate(&hd))
        throw blas_ex("cublas context creation failed.\n");
      isInit = true;
    }
  }

  void free_cublas () {
    if (isInit) {
      if (CUBLAS_STATUS_SUCCESS != cublasDestroy(hd))
        throw blas_ex("cublas context release failed (not yet created at all?).\n");
      isInit = false;
    }
  }

  void assert_succeed (cublasStatus_t sta) {
    if (CUBLAS_STATUS_SUCCESS != sta )
      throw blas_ex("cublas computation failed.\n");
    // TODO: more informative message!
  }

private:
  bool isInit;
  cublasHandle_t hd;

private:
  cublasMon() : isInit(false) {};

  cublasMon      (cublasMon const&); 
  void operator= (cublasMon const&); 

};


//// Impl of exception
blas_ex::blas_ex(const char* msg)
  : runtime_error(msg)
{
}


//// Impl of release_cublas_context
void release_cublas_context ()
{
  cublasMon::get_this().free_cublas();
}


//// Impl of the gpu_* functions
void cu_AxBtoC(const matw &A, const matw &B, matw &C, bool isOverWrite)
{
  // A: [M, K], B: [K, N]
  int M = static_cast<int>(A.H); // assert (M == C.H)
  int K = static_cast<int>(A.W); // assert (K == B.H)
  int N = static_cast<int>(B.W); // assert (N == C.W)

  float alpha = 1.0;
  float beta = isOverWrite? 0.0 : 1.0;

  cublasStatus_t st = cublasSgemm(
    cublasMon::get_this().get_cublasHandler(),
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    (float*)A.beg, M,
    (float*)B.beg, K,
    &beta,
    (float*)C.beg, M);

  cublasMon::get_this().assert_succeed(st);
  return;
}

void cu_ATxBtoC(const matw &A, const matw &B, matw &C, bool isOverWrite)
{
  // A [K, M], B: [K, N], C: [M, N]
  int M = static_cast<int>(A.W);
  int K = static_cast<int>(A.H);
  int N = static_cast<int>(B.W);

  int ldA = K;
  int ldB = K;
  int ldC = M;

  float alpha = 1.0;
  float beta = isOverWrite? 0.0 : 1.0;

  cublasStatus_t st = cublasSgemm(
    cublasMon::get_this().get_cublasHandler(),
    CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    (float*)A.beg, ldA,
    (float*)B.beg, ldB,
    &beta,
    (float*)C.beg, ldC);

  cublasMon::get_this().assert_succeed(st);
  return;
}

void cu_AxBTtoC(const matw &A, const matw &B, matw &C, bool isOverWrite)
{
  // A [M, K], B: [N, K], C: [M, N]
  int M = static_cast<int>(A.H);
  int K = static_cast<int>(A.W);
  int N = static_cast<int>(B.H);

  int ldA = M;
  int ldB = N;
  int ldC = M;

  float alpha = 1.0;
  float beta = isOverWrite? 0.0 : 1.0;

  cublasStatus_t st = cublasSgemm(
    cublasMon::get_this().get_cublasHandler(),
    CUBLAS_OP_N, CUBLAS_OP_T,
    M, N, K,
    &alpha,
    (float*)A.beg, ldA,
    (float*)B.beg, ldB,
    &beta,
    (float*)C.beg, ldC);

  cublasMon::get_this().assert_succeed(st);
  return;
}

#else
#error WITH_GPUARRAY macro must be defined in order to compile this file.
#endif
