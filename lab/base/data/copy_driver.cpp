#include <cstdint>
#include <iostream>

extern "C" {
  void copy_c_0( int32_t const * __restrict a,
                 int32_t       * __restrict b );

  void copy_c_1( int64_t                    n,
                 int32_t const * __restrict a,
                 int32_t       * __restrict b );

  void copy_asm_0( int32_t const * a,
                   int32_t       * b );

  void copy_asm_1( int64_t         n,
                   int32_t const * a,
                   int32_t       * b );
}

void init( int64_t   n,
           int32_t * a,
           int32_t * b ) {
  for( int64_t i = 0; i < n; ++i ) {
    a[i] = i;
    b[i] = 0;
  }
}

void check( int64_t   n,
            int32_t * a,
            int32_t * b ) {
  bool l_ok = true;
  for( int64_t i = 0; i < n; ++i ) {
    if( a[i] != b[i] ) {
      l_ok = false;
      break;
    }
  }
  if( l_ok ) {
    std::cout << "copy succeeded" << std::endl;
  } else {
    std::cout << "copy failed" << std::endl;
  }
}

int main() {
  int64_t l_n = 25;

  // allocate memory
  int32_t * l_a = new int32_t[l_n];
  int32_t * l_b = new int32_t[l_n];

  /*
   * copy_c_0
   */
  std::cout << "copy_c_0: ";
  init( l_n,
        l_a,
        l_b );

  copy_c_0( l_a,
            l_b );

  check( 7,
         l_a,
         l_b );

  /*
   * copy_c_1
   */
  std::cout << "copy_c_1: ";
  init( l_n,
        l_a,
        l_b );

  copy_c_1( l_n,
            l_a,
            l_b );

  check( l_n,
         l_a,
         l_b );

  /*
   * copy_asm_0
   */
  std::cout << "copy_asm_0: ";
  init( l_n,
        l_a,
        l_b );

  copy_asm_0( l_a,
              l_b );

  check( 7,
         l_a,
         l_b );

  /*
   * copy_asm_1
   */
  std::cout << "copy_asm_1: ";
  init( l_n,
        l_a,
        l_b );

  copy_asm_1( l_n,
              l_a,
              l_b );
  check( l_n,
         l_a,
         l_b );

  // free memory
  delete[] l_a;
  delete[] l_b;

  return EXIT_SUCCESS;
}