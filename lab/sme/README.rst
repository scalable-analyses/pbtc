Scalable Matrix Extension
=========================

Execution Throughput and Latency
--------------------------------

This section uses microbenchmarks to measure the execution throughput and latency of Matrix Outer Product Accumulate (MOPA) instructions.

.. admonition:: Tasks

   1. Microbenchmark the execution throughput of the following instructions:

      * FP32 FMOPA (non-widening).
      * BFMOPA (widening).
   
   2. Microbenchmark the execution latency of FP32 FMOPA (non-widening). Consider dependencies on the destination ZA tile.

Microkernel
-----------

This section implements an FP32 SME microkernel for matrix-matrix multiplications.
The microkernel uses a 32x32 accumulator and is wrapped in the ``matmul_32_32_1`` function, which has the following signature:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_32_32_1( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );

.. admonition:: Tasks

   1. Implement an SME microkernel that computes C+=AB for M=32, N=32, and K=1.
      Wrap your microkernel in the ``matmul_32_32_1`` function.
   
   2. Test and optimize your microkernel. Report its performance in GFLOPS.

Loops
-----

This section adds loops around the microkernel to implement matmuls on larger matrices.
First, we add a loop over K and write an extended kernel with the following function signature:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_32_32_512( float   const * a,
                          float   const * b,
                          float         * c,
                          int64_t         ld_a,
                          int64_t         ld_b,
                          int64_t         ld_c );

Next, we add a loop over M to implement:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_512_32_512( float   const * a,
                           float   const * b,
                           float         * c,
                           int64_t         ld_a,
                           int64_t         ld_b,
                           int64_t         ld_c );

Finally, we add a loop over N to implement:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_512_512_512( float   const * a,
                            float   const * b,
                            float         * c,
                            int64_t         ld_a,
                            int64_t         ld_b,
                            int64_t         ld_c );

.. admonition:: Tasks

   1. Loop over K: Implement a kernel that computes C+=AB for M=32, N=32 and K=512.
      Wrap your kernel in the ``matmul_32_32_512`` function.

   2. Loop over M: Implement a kernel that computes C+=AB for M=512, N=32 and K=512.
      Wrap your kernel in the ``matmul_512_32_512`` function.

   3. Loop over N: Implement a kernel that computes C+=AB for M=512, N=512 and K=512.
      Wrap your kernel in the ``matmul_512_512_512`` function.

   4. Test and optimize the kernels. Report your performance in GFLOPS.