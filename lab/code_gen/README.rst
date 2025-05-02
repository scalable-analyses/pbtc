Code Generation
===============

This chapter develops software that generates batch-reduce matrix-matrix multiplications (BRGEMMs).
As outlined in the file `Brgemm.h <data/Brgemm.h>`_ in the ``data`` directory, we use the C++ namespace ``mini_jit::Brgemm`` for all development.
Use the ``generate`` function as a common entry point for all BRGEMM code generation:

.. code-block:: C++

  /**
   * @brief Generate a kernel for batch-reduce matrix multiplication.
   * @param m number of rows in A and C.
   * @param n number of columns in B and C.
   * @param k number of columns in A and rows in B.
   * @param br_size batch-reduce size.
   * @param trans_a 0 if A is stored in column-major order, 1 if A is stored in row-major order.
   * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
   * @param trans_c 0 if C is stored in column-major order, 1 if C is stored in row-major order.
   * @param dtype data type of the matrices.
   * @return error_t::success on success, another error_t value otherwise.
   **/
  error_t generate( uint32_t m,
                    uint32_t n,
                    uint32_t k,
                    uint32_t br_size,
                    uint32_t trans_a,
                    uint32_t trans_b,
                    uint32_t trans_c,
                    dtype_t  dtype );

All generated kernels have the same function signature and can be obtained by calling the ``get_kernel`` function:

.. code-block:: C++

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

Microkernel
-----------
This section develops software that can generate a microkernel and a loop over the matrix dimension K.

.. admonition:: Tasks

   1. Start implementing the ``generate`` function. Support only the single setting of an FP32 Neon microkernel that computes C+=AB for column-major matrices and M=16, N=6, and K=1.
      Return an appropriate error code if the parameters of the function differ from this setting.
   
   2. Add support for the ``k`` parameter by generating a K loop around the microkernel.

   3. Test the kernel generation. Report performance in GFLOPS.