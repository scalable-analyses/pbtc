#include <stdint.h>

void copy_c_0( int32_t * __restrict a,
               int32_t * __restrict b ) {
  b[0] = a[0];
  b[1] = a[1];
  b[2] = a[2];
  b[3] = a[3];
  b[4] = a[4];
  b[5] = a[5];
  b[6] = a[6];
}

void copy_c_1( int64_t              n,
               int32_t * __restrict a,
               int32_t * __restrict b ) {
  for( uint64_t i = 0; i < n; ++i ) {
    b[i] = a[i];
  }
}