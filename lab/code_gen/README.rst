Code Generation
===============

BRGEMM Primitive
----------------

This section develops software that generates batch-reduce matrix-matrix multiplications (BRGEMMs).
As outlined in the file `Brgemm.h <data/Brgemm.h>`_ in the ``data`` directory, we use the C++ namespace ``mini_jit::Brgemm`` for all development.
Use the ``generate`` function as a common entry point for all BRGEMM code generation:

.. code-block:: C++

   /**
    * @brief Generate a kernel for batch-reduce matrix multiplication.
    * @param m       Number of rows in A and C.
    * @param n       Number of columns in B and C.
    * @param k       Number of columns in A and rows in B.
    * @param br_size Batch-reduce size.
    * @param trans_a 0 if A is stored in column-major order, 1 if A is stored in row-major order.
    * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
    * @param trans_c 0 if C is stored in column-major order, 1 if C is stored in row-major order.
    * @param dtype   Data type of the matrices.
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
    * - a:           Pointer to first of a batch of A matrices.
    * - b:           Pointer to first of a batch of B matrices.
    * - c:           Pointer to C matrix.
    * - ld_a:        Leading dimension of A.
    * - ld_b:        Leading dimension of B.
    * - ld_c:        Leading dimension of C.
    * - br_stride_a: Stride (in elements, not bytes) between A matrices.
    * - br_stride_b: Stride (in elements, not bytes) between B matrices.
    */
   using kernel_t = void (*)( void    const * a,
                              void    const * b,
                              void          * c,
                              int64_t         ld_a,
                              int64_t         ld_b,
                              int64_t         ld_c,
                              int64_t         br_stride_a,
                              int64_t         br_stride_b );

  /**
   * @brief Get the generated kernel: C += sum_i(A_i * B_i).
   * @return pointer to the generated kernel.
   **/
  kernel_t get_kernel() const;

Microkernel
^^^^^^^^^^^
This section adds support for generating a microkernel and a loop over the matrix dimension K.

.. admonition:: Tasks

   1. Start implementing the ``generate`` function. Support only the single setting of an FP32 Neon microkernel that computes C+=AB for column-major matrices and M=16, N=6, and K=1.
      Return an appropriate error code if the parameters of the function differ from this setting.
   
   2. Add support for the ``k`` parameter by generating a K loop around the microkernel.

   3. Test the kernel generation. Report performance in GFLOPS.

GEMM
^^^^
This section extends the code generation software by supporting many choices for the dimensions M, N and K.
Specifically, support should include at least all M-N-K combinations for the following ranges:

* 1≤M≤1024
* 1≤N≤1024
* 1≤K≤2048

.. admonition:: Tasks

   1. Extend the implementation of the ``generate`` function to support all M-N-K combinations for C+=AB as specified above. Assume that all matrices are in column-major format.

   2. Verify your kernel generation by comparing to a reference implementation for 1≤M≤64, 1≤N≤64 and K∈[1,16,32,64,128], and by setting lda=M, ldb=K, ldc=M.

   3. Verify the kernel generation in cases where lda>M, ldb>K or ldc>M.

   4. Benchmark the performance of your generated kernels and report the measured performance for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], lda=M, ldb=K and ldc=M. Use a CSV format for output. Follow the structure of the example file `data/perf.csv <data/perf.csv>`_. Report the arithmetic mean performance of all settings in GFLOPS.

Batch-Reduce GEMM
^^^^^^^^^^^^^^^^^

This section extends the code generation with support for a batch-reduce dimension. We assume that 1≤br_size≤1024.

.. admonition:: Tasks

  1. Extend the implementation of the ``generate`` function to support batch-reduce GEMMs: C+=∑AᵢBᵢ. Assume that all matrices are in column-major format.

  2. Verify your generated kernels against a reference implementation.

  3. Benchmark the performance of your generated kernels and report the measured performance for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], br_size=16, br_stride_a=M*K, br_stride_b=K*N, lda=M, ldb=K, and ldc=M. Report the arithmetic mean performance of all settings in GFLOPS.

Unary Primitives
----------------
This section develops software that generates kernels  instantiating unary primitives of the form B:=op(A).
As described in the `Unary.h <data/Unary.h>`_ file in the ``data`` directory, the ``mini_jit::Unary`` C++ namespace is used for all development.
Use the ``generate`` function as a common entry point for all unary primitives:

.. code-block:: C++

   /**
    * @brief Generate a kernel for a unary primitive.
    * @param m       Number of rows in A and B.
    * @param n       Number of columns in A and B.
    * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
    * @param dtype   Data type of the matrices.
    * @param ptype   Primitive type.
    * @return error_t::success on success, another error_t value otherwise.
    **/
   error_t generate( uint32_t m,
                     uint32_t n,
                     uint32_t trans_b,
                     dtype_t  dtype,
                     ptype_t  ptype );

All generated kernels have the same function signature and can be obtained by calling the ``get_kernel`` function:

.. code-block:: C++

   /*
    * Kernel type.
    * The kernel is a function that takes the following parameters:
    * - a:    Pointer to column-major matrix A, nullptr if zero kernel.
    * - b:    Pointer to matrix B.
    * - ld_a: Leading dimension of A.
    * - ld_b: Leading dimension of B.
    */
   using kernel_t = void (*)( void    const * a,
                              void          * b,
                              int64_t         ld_a,
                              int64_t         ld_b );

   /**
    * @brief Get the generated kernel: B := op(A).
    * @return pointer to the generated kernel.
    **/
   kernel_t get_kernel() const;

Our code generation software should support at least the following dimension sizes:

* 1≤M≤2048
* 1≤N≤2048

Zero Primitive
^^^^^^^^^^^^^^
We begin the development of code generation for unary primitives by implementing support for the zero primitive: B:=0.
The zero primitive simply sets the output matrix to zero, so instantiating kernels simply ignore the input matrix A.

.. admonition:: Tasks

   1. Begin implementing the ``mini_jit::Unary::generate`` function by adding support for the zero primitive.
   2. Test and benchmark representative kernels. Include at least settings with M=N=50, M=N=64, M=N=512 and M=N=2048. Report performance in GiB/s.

Identity Primitive
^^^^^^^^^^^^^^^^^^
This section extends our code generation of unary primitives by adding support for transposition.
With ``mini_jit::Unary``, a transposition corresponds to a kernel that instantiates the ``identity`` primitive with the ``trans_b`` parameter set to ``1``.
For example, suppose we need a kernel that transposes a 32x64 FP32 matrix.
In this case, we would call the ``generate`` function with the following parameters:

.. code-block:: C++

   generate( 32,                  // m
             64,                  // n
             1,                   // trans_b
             dtype_t::fp32,       // dtype
             ptype_t::identity ); // ptype

.. admonition:: Tasks

   1. Extend the implementation of the ``mini_jit::Unary::generate`` function to support the identity primitive.
   2. Test and benchmark representative kernels. Include at least settings with M=N=50, M=N=64, M=N=512 and M=N=2048. Report performance in GiB/s.

ReLU Primitive
^^^^^^^^^^^^^^
This section adds support for the Rectified Linear Unit (ReLU).

.. admonition:: Tasks

   1. Extend the implementation of the ``mini_jit::Unary::generate`` function to support the ReLU primitive.
   2. Test and benchmark representative kernels. Include at least settings with M=N=50, M=N=64, M=N=512 and M=N=2048. Report performance in GiB/s.