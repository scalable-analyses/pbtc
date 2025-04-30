#ifndef MINI_JIT_BRGEMM_H
#define MINI_JIT_BRGEMM_H

#include <cstdint>

namespace mini_jit {
  class Brgemm;
}

class mini_jit::Brgemm {
  public:
    /// data type
    enum class dtype_t : uint32_t {
      fp32 = 0,
      fp64 = 1
    };

    /// error codes
    enum class error_t : int32_t {
      success = 0
    };

  /**
   * @brief Generate a kernel for batch-reduce matrix multiplication.
   * @param m number of rows in A and C.
   * @param n number of columns in B and C.
   * @param k number of columns in A and rows in B.
   * @param br_size batch-reduce size.
   * @param dtype data type of the matrices.
   * @return error_t::success on success, another error_t value otherwise.
   **/
  error_t generate( uint32_t m,
                    uint32_t n,
                    uint32_t k,
                    uint32_t br_size,
                    dtype_t  dtype );


  /*
   * Kernel type.
   * The kernel is a function that takes the following parameters:
   * - a: pointer to first column-major A matrix.
   * - b: pointer to first column-major B matrix.
   * - c: pointer to first column-major C matrix.
   * - lda: leading dimension of A.
   * - ldb: leading dimension of B.
   * - ldc: leading dimension of C.
   * - br_stride_a: stride between two A matrices (in elements, not bytes).
   * - br_stride_b: stride between two B matrices (in elements, not bytes).
   */
  using kernel_t = void (*)( void    const * a,
                             void    const * b,
                             void          * c,
                             int64_t         lda,
                             int64_t         ldb,
                             int64_t         ldc,
                             int64_t         br_stride_a,
                             int64_t         br_stride_b );

  /**
   * @brief Get the generated kernel: C += sum_i(A_i * B_i).
   * @return pointer to the generated kernel.
   **/
  kernel_t get_kernel() const;
};

#endif