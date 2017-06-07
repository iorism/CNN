#include "staticMem.h"
#include "_staticMem.h"


//// Static buffer
static buf_cpu_t bufZeros_cpu;
static buf_cpu_t bufOnes_cpu;


//// Impl of buf_cpu_t
void buf_cpu_t::realloc( size_t _nelem )
{
  dealloc();
  beg = (float*) malloc( _nelem * sizeof(float) );
  if (beg == 0) throw sm_ex("staticMem: Out of CPU memory.\n");
  nelem = _nelem;
}

void buf_cpu_t::dealloc()
{
  if (beg != 0) free( (void*) beg );
  beg = 0;
  nelem = 0;
}


//// Impl of cpu interface
float* sm_zeros_cpu (size_t nelem) 
{
  if ( bufZeros_cpu.is_need_realloc(nelem) ) {
    bufZeros_cpu.realloc(nelem);
    for (int i = 0; i < nelem; ++i) bufZeros_cpu.beg[i] = 0.0;
  }
  return bufZeros_cpu.beg;
}

float* sm_ones_cpu (size_t nelem)
{
  if ( bufOnes_cpu.is_need_realloc(nelem) ) {
    bufOnes_cpu.realloc(nelem);
    for (int i = 0; i < nelem; ++i) bufOnes_cpu.beg[i] = 1.0;
  }
  return bufOnes_cpu.beg;
}

void   sm_release_cpu ()
{
  bufZeros_cpu.dealloc();
  bufOnes_cpu.dealloc();
}