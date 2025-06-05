Tensor Operations
=================

Backend
-------

This section implements a backend for binary tensor contractions and unary tensor permutations.
The structure of the backend is outlined in the file `TensorOperation.h <data/TensorOperation.h>`_ in the ``data`` directory. 
The backend performs the provided tensor operation exactly as defined by the interface and does not optimize it.
Contractions are executed as recursive loops over small GEMM or Batch-Reduce GEMM (BRGEMM) kernels.
Permutations are executed as recursive loops over small transposition kernels.

User Interface
^^^^^^^^^^^^^^

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

.. code-block:: C++

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
^^^^^^^^^^^^^^^^^^^^^^^^

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

Shared Memory Parallelization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the shared memory domain, loops can be parallelized at any point within the nested loop structure.
However, to simplify the implementation, we only parallelize the outermost loops.
In other words, we do not parallelize loops that are nested inside sequential loops.

To support an arbitrary number of parallel loops, a simple implementation could fuse them and use division and modulo operations to reconstruct the indices in the original loops.
The following high-level code example shows one way to achieve this:

.. code-block:: C++


    /**
     * General-purpose loop implementation featuring first and last touch operations with OMP parallelization.
     *
     * @param ptr_in0      Pointer to the first input tensor's data.
     * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
     * @param ptr_out      Pointer to the output tensor's data.
     * @param first_access True if first time accessing data of output tensor.
     * @param last_access  True if last time accessing data of output tensor.
     **/
    void execute_iter_parallel( char const * ptr_in0,
                                char const * ptr_in1,
                                char       * ptr_out,
                                bool         first_access,
                                bool         last_access ) {
    #pragma omp parallel for
      for( int64_t l_it_all = 0; l_it_all < m_size_parallel_loops; l_it_all++ ) {
        int64_t l_it_remaining = l_it_all;
        for( int64_t l_id_loop = m_num_parallel_loops - 1; l_id_loop >= 0; l_id_loop-- ) {
          // calculate loop index l_it for loop l_id_loop
          int64_t l_it   = l_it_remaining % m_loop_sizes[l_id_loop];
          l_it_remaining = l_it_remaining / m_loop_sizes[l_id_loop];

          // derive if this is first or last access to the output block

          // update pointer with strides
        }
        // call non parallel loops or kernel
      }
    }

.. admonition:: Task

  Implement the function ``execute_iter_parallel``, which parallelizes a binary tensor contraction in the shared memory domain.

Optimization Passes
-------------------
This section employs various optimization passes to enhance the performance of tensor operations.
Having an Intermediate Representation (IR) that allows for dimension reordering, splitting, and fusion as transformations is advantageous for implementing optimization passes.
Passes that could use these transformations include the following:

#. Dimension splitting
#. Dimension fusion
#. Dimension reordering
#. Primitive identification
#. Shared memory parallelization

.. list-table:: Matrix multiplication example.
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Value
   * - dim_types
     - (    M,    N,    K )
   * - exec_types
     - (  Seq,  Seq,  Seq )
   * - dim_sizes
     - ( 1600, 1600, 1600 )
   * - strides_in0
     - (    1,    0, 1600 )
   * - strides_in1
     - (    0, 1600,    1 )
   * - strides_out
     - (    1, 1600,    0 )

.. list-table:: Tensor contraction example.
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Value
   * - dim_types
     - (   M,    M,     N,    N,     K,    K )
   * - exec_types
     - ( Seq,  Seq,   Seq,  Seq,   Seq,  Seq )
   * - dim_sizes
     - (  64,   25,    64,   25,    64,   25 )
   * - strides_in0
     - (  25,    1,     0,    0, 40000, 1600 )
   * - strides_in1
     - (   0,    0, 40000, 1600,    25,    1 )
   * - strides_out
     - (  25,    1, 40000, 1600,     0,    0 )


.. admonition:: Tasks

   1. Develop an IR that supports transformations such as dimension reordering, dimension splitting and fusing dimensions.
   2. Implement optimization passes. At a minimum, support primitive identification and shared memory parallelization.
   3. Lower the optimized IR code to your tensor operation backend. Verify the correctness of the optimizations.
   4. Benchmark the performance of your implementation for the above matrix multiplication and tensor contraction examples. Report the measured performance in GFLOPS.
   5. Demonstrate the capabilities of your optimization passes using your own examples.

Unary Operations
----------------

Unary tensor operations, such as permuting a tensor's dimensions, can be implemented using the same interface as binary operations.
In a unary operation, all of the input tensor dimensions are present in the output tensors.
Therefore, we use the dimension type ``dim_t::c`` for all dimensions.
The ``execute`` function ignores the second input pointer, ``ptr_in1``, for unary tensor operations.
Thus, we pass ``nullptr`` for unary executions.

.. admonition:: Tasks

   1. Extend the tensor operation backend to support unary tensor operations.
   2. Implement primitive identification and shared memory parallelization optimization passes for unary tensor operations.
   3. Verify the correctness of your implementation against a reference implementation.