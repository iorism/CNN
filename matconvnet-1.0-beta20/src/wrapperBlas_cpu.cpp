#include "wrapperBlas.h"
#include <blas.h>


namespace {

template<bool TA, bool TB, bool isOverWrite>
void CeqAxB_tmpl(const matw &A, const matw &B, matw &C)
{


}

}

void AxBtoC(const matw &A, const matw &B, matw &C, bool isOverWrite)
{
  // A: [M, K], B: [K, N]
  ptrdiff_t M = A.H; // assert (M == C.H)
  ptrdiff_t K = A.W; // assert (K == B.H)
  ptrdiff_t N = B.W; // assert (N == C.W)

  float alpha = 1.0;
  float beta = isOverWrite? 0.0 : 1.0;

  sgemm(
    "n", "n",
    &M, &N, &K,
    &alpha,
    (float*)A.beg, &M,
    (float*)B.beg, &K,
    &beta,
    (float*)C.beg, &M);

  return;
}

void ATxBtoC(const matw &A, const matw &B, matw &C, bool isOverWrite)
{
  // A [K, M], B: [K, N], C: [M, N]
  ptrdiff_t M = A.W;
  ptrdiff_t K = A.H;
  ptrdiff_t N = B.W;

  ptrdiff_t ldA = K;
  ptrdiff_t ldB = K;
  ptrdiff_t ldC = M;

  float alpha = 1.0;
  float beta = isOverWrite? 0.0 : 1.0;

  sgemm(
    "t", "n",
    &M, &N, &K,
    &alpha,
    (float*)A.beg, &ldA,
    (float*)B.beg, &ldB,
    &beta,
    (float*)C.beg, &ldC);

  return;
}

void AxBTtoC(const matw &A, const matw &B, matw &C, bool isOverWrite)
{
  // A [M, K], B: [N, K], C: [M, N]
  ptrdiff_t M = A.H;
  ptrdiff_t K = A.W;
  ptrdiff_t N = B.H;

  ptrdiff_t ldA = M;
  ptrdiff_t ldB = N;
  ptrdiff_t ldC = M;

  float alpha = 1.0;
  float beta = isOverWrite? 0.0 : 1.0;

  sgemm(
    "n", "t",
    &M, &N, &K,
    &alpha,
    (float*)A.beg, &ldA,
    (float*)B.beg, &ldB,
    &beta,
    (float*)C.beg, &ldC);

  return;
}
