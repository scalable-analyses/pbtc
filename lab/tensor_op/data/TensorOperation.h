#ifndef EINSUM_BACKEND_TENSOR_OPERATION_H
#define EINSUM_BACKEND_TENSOR_OPERATION_H

#include <cstdint>
#include <span>

namespace einsum {
  namespace backend {
    class TensorOperation;
  }
}

class einsum::backend::TensorOperation {
  public:
    /// execution type
    enum class exec_t : uint32_t {
      seq    = 0, 
      prim   = 1,
      shared = 2, 
    };

    /// primitive type
    enum class prim_t : uint32_t {
      zero   =  0,
      copy   =  1,
      relu   =  2,
      gemm   =  3,
      brgemm =  4,
      none      = 99
    };

    /// dimension type
    enum class dim_t : uint32_t {
      c         = 0, 
      m         = 1, 
      n         = 2, 
      k         = 3, 
      undefined = 99
    };

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

    /**
     * Execute the tensor operation.
     *
     * @param tensor_in0 First input tensor.
     * @param tensor_in1 Second input tensor (use nullptr if unary).
     * @param tensor_out Output tensor.
     **/
    void execute( void const * tensor_in0,
                  void const * tensor_in1,
                  void       * tensor_out );
    
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
                       bool            last_access );
};

#endif