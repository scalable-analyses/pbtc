Tensor Operation Backend
========================

This section implements a backend for binary tensor contractions and unary tensor permutations.
The structure of the backend is outlined in the file `TensorOperation.h <data/TensorOperation.h>`_ in the ``data`` directory. 
The backend performs the provided tensor operation exactly as defined by the interface and does not optimize it.
Contractions are executed as recursive loops over small GEMM or Batch-Reduce GEMM (BRGEMM) kernels.
Permutations are executed as recursive loops over small transposition kernels.

User Interface
--------------

The user interface of the backend is given by the ``setup`` function that has the following signature:

.. code-block:: C++

    /**
     * Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param dtype             Datatype of all tensor elements.
     * @param prim_first_touch  Type of the first touch primitive.
     * @param prim_main         Type of the main primitive.
     * @param prim_last_touch   Type of the last touch primitive.
     * @param dim_types         Dimension type of the loops (c, m, n, or k).
     * @param exec_types        Execution type of the loops (seq, shared, or prim).
     * @param dim_sizes         Sizes of the dimensions.
     * @param strides_in0       Strides of the first input tensor.
     * @param strides_in1       Strides of the second input tensor (ignored if unary).
     * @param strides_out       Strides of the output tensor.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    error_t setup( dtype_t                    dtype,
                   prim_t                     prim_first_touch,
                   prim_t                     prim_main,
                   prim_t                     prim_last_touch,
                   std::span< const dim_t >   dim_types,
                   std::span< const exec_t >  exec_types,
                   std::span< const int64_t > dim_sizes,
                   std::span< const int64_t > strides_in0,
                   std::span< const int64_t > strides_in1,
                   std::span< const int64_t > strides_out );



.. admonition:: Task

   Begin implementing the ``setup`` function of the class ``einsum::backend::TensorOperation`` for binary tensor contractions.
   Parse the configuration parameters passed to the function and generate the corresponding (BR)GEMM kernel at runtime.

Recursive Loops Over Primitives
-------------------------------
Similarly to the following code example, we can define a recursive contraction function for an arbitrary number of loops:

.. code-block:: C

    /**
     * General-purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param id_loop      Dimension id of the loop which is executed.
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_iter( int64_t         id_loop,
                       char    const * ptr_in0,
                       char    const * ptr_in1,
                       char          * ptr_out,
                       bool            first_access,
                       bool            last_access ) {
      int64_t l_size  = m_loop_sizes[id_loop];

      for( int64_t l_it = 0; l_it < l_size; l_it++ ) {
         // derive if this is first or last access to the output block

         // update pointer with strides

         if( id_loop + 1 < m_id_first_primitive_loop ) {
            // recursive function call
         }
         else {
            // call first touch kernel if necessary

            // call main kernel
            
            // call last touch kernel if necessary
         }
      }
   }

.. admonition:: Tasks

   1. Implement the ``execute`` function of the ``einsum::backend::TensorOperation`` class using recursive loops over primitives.
      Limit your implementation to single-threaded execution.

   2. Verify your implementation against a reference implementation.

Performance Benchmarking
------------------------

.. list-table:: Tensor contraction using the GEMM primitive.
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Value
   * - dtype
     - FP32
   * - prim_first_touch
     - None
   * - prim_main
     - GEMM
   * - prim_last_touch
     - None
   * - dim_types
     - (     M,    N,    K,    M,    N,    K )
   * - exec_types
     - (   Seq,  Seq,  Seq, Prim, Prim, Prim )
   * - dim_sizes
     - (    32,   32,    8,   32,   32,   32 )
   * - strides_in0
     - (  8192,    0, 1024,    1,    0,   32 )
   * - strides_in1
     - (     0, 8192, 1024,    0,   32,    1 )
   * - strides_out
     - ( 32768, 1024,    0,    1,   32,    0 )

|

.. list-table:: Tensor contraction using the BRGEMM primitive.
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Value
   * - dtype
     - FP32
   * - prim_first_touch
     - None
   * - prim_main
     - BRGEMM
   * - prim_last_touch
     - None
   * - dim_types
     - (     M,    N,    K,    M,    N,    K )
   * - exec_types
     - (   Seq,  Seq, Prim, Prim, Prim, Prim )
   * - dim_sizes
     - (    32,   32,    8,   32,   32,   32 )
   * - strides_in0
     - (  8192,    0, 1024,    1,    0,   32 )
   * - strides_in1
     - (     0, 8192, 1024,    0,   32,    1 )
   * - strides_out
     - ( 32768, 1024,    0,    1,   32,    0 )

|

.. list-table:: Tensor contraction using the Zero, BRGEMM and ReLU primitives.
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Value
   * - dtype
     - FP32
   * - prim_first_touch
     - Zero
   * - prim_main
     - BRGEMM
   * - prim_last_touch
     - ReLU
   * - dim_types
     - (     M,    N,    K,    M,    N,    K )
   * - exec_types
     - (   Seq,  Seq, Prim, Prim, Prim, Prim )
   * - dim_sizes
     - (    32,   32,    8,   32,   32,   32 )
   * - strides_in0
     - (  8192,    0, 1024,    1,    0,   32 )
   * - strides_in1
     - (     0, 8192, 1024,    0,   32,    1 )
   * - strides_out
     - ( 32768, 1024,    0,    1,   32,    0 )

.. admonition:: Tasks

   1. Benchmark the performance of your implementation for the above examples. Report the measured performance in GFLOPS.

   2. Design your own setups. Which setups achieve a high performance and which setups are slow?