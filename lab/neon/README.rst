Neon
====

Execution Throughput and Latency
--------------------------------

This section microbenchmarks the execution throughput and latency of FP32 Neon instructions.

.. admonition:: Tasks

   1. Microbenchmark the execution throughput of the following instructions:

      * FMLA (vector) with arrangement specifier ``4S``.
      * FMLA (vector) with arrangement specifier ``2S``.
      * FMADD (scalar), single-precision variant.
   
   2. Microbenchmark the execution latency of FMLA (vector) with arrangement specifier ``4S``. Consider the following two cases:

      * Dependency on one of the source registers.
      * Dependency on the destination register only.

Microkernel
-----------

This section implements an FP32 Neon microkernel for matrix-matrix multiplications.
The microkernel uses a 16x6 accumulator block and is wrapped in the ``matmul_16_6_1`` function with the following C function signature:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to column-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_16_6_1( float   const * a,
                       float   const * b,
                       float         * c,
                       int64_t         ld_a,
                       int64_t         ld_b,
                       int64_t         ld_c );

.. admonition:: Tasks

   1. Implement a Neon microkernel that computes C+=AB for M=16, N=6, and K=1.
      Wrap your microkernel in the ``matmul_16_6_1`` function.
   
   2. Test and optimize your microkernel. Report its performance in GFLOPS.

Loops
-----

This section adds loops around the microkernel to implement matmuls on larger matrices.
We start by adding a loop over K and writing an extended kernel with the following function signature:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a   Pointer to column-major matrix A.
    * @param b   Pointer to column-major matrix B.
    * @param c   Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_16_6_64( float   const * a,
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
    * @param b    Pointer to column-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_64_6_64( float   const * a,
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
    * @param b    Pointer to column-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_64_48_64( float   const * a,
                         float   const * b,
                         float         * c,
                         int64_t         ld_a,
                         int64_t         ld_b,
                         int64_t         ld_c );

.. admonition:: Tasks

   1. Loop over K: Implement a kernel that computes C+=AB for M=16, N=6 and K=64.
      Wrap your kernel in the ``matmul_16_6_64`` function.

   2. Loop over M: Implement a kernel that computes C+=AB for M=64, N=6 and K=64.
      Wrap your kernel in the ``matmul_64_6_64`` function.

   3. Loop over N: Implement a kernel that computes C+=AB for M=64, N=48 and K=64.
      Wrap your kernel in the ``matmul_64_48_64`` function.

   4. Test and optimize the kernels. Report your performance in GFLOPS.

SIMD Lanes
----------

This section considers matrix-matrix multiplications, that require instructions where only a subset of SIMD lanes are active.
The first kernel has the following function signature:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to column-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_14_6_64( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );

The second kernel has this function signature:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to column-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_15_6_64( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );

.. admonition:: Tasks

   1. Implement a kernel that computes C+=AB for M=14, N=6 and K=64.
      Wrap your kernel in the ``matmul_14_6_64`` function.

   2. Implement a kernel that computes C+=AB for M=15, N=6 and K=64.
      Wrap your kernel in the ``matmul_15_6_64`` function.

   3. Test and optimize the kernels. Report your performance in GFLOPS.

Accumulator Block Shapes
------------------------
This section considers a matrix-matrix multiplication where a high-performance implementation may require accumulator blocks with different shapes.
The kernel has the following function signature:

.. code-block:: C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to column-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void matmul_64_64_64( float   const * a,
                         float   const * b,
                         float         * c,
                         int64_t         ld_a,
                         int64_t         ld_b,
                         int64_t         ld_c );

.. admonition:: Tasks

   1. Implement a kernel that computes C+=AB for M=64, N=64 and K=64.
      Wrap your kernel in the ``matmul_64_64_64`` function.

   2. Test and optimize the kernel. Report your performance in GFLOPS.

Batch-Reduce GEMM
-----------------
This section considers a batch-reduce matrix-matrix multiplication that has a fourth dimension in addition to the known M, N, and K dimensions.
The kernel takes as input a batch of matrices Aᵢ and Bᵢ and computes C+=∑AᵢBᵢ.
The kernel has the following function signature:

.. code-block:: C

   /**
    * @brief Batch-reduce GEMM that computes: C+=sum(Ai*Bi) over a batch.
    * @param a           Pointer to first of a batch of column-major A matrices.
    * @param b           Pointer to first of a batch of column-major B matrices.
    * @param c           Pointer to column-major C matrix.
    * @param ld_a        Leading dimension of A.
    * @param ld_b        Leading dimension of B.
    * @param ld_c        Leading dimension of C.
    * @param br_stride_a Stride (in elements, not bytes) between A matrices.
    * @param br_stride_b Stride (in elements, not bytes) between B matrices.
    **/
   void matmul_64_48_64_16( float   const * a,
                            float   const * b,
                            float         * c,
                            int64_t         ld_a,
                            int64_t         ld_b,
                            int64_t         ld_c,
                            int64_t         br_stride_a,
                            int64_t         br_stride_b );

The parameter ``br_stride_a`` specifies the stride between two consecutive Aᵢ matrices.
The parameter ``br_stride_b`` specifies the stride between two consecutive Bᵢ matrices.
For example, if ``br_stride_a`` is 4096, the two matrices A₀ and A₁ are 4096 FP32 values or  16 KiB apart.

.. admonition:: Tasks

   1. Implement a kernel that computes C+=∑AᵢBᵢ for M=64, N=48 and K=64 and a batch-reduce dimension size of 16.
      Wrap your kernel in the ``matmul_64_48_64_16`` function.

   2. Test and optimize the kernel. Report your performance in GFLOPS.