#include "MatmulLibxsmm.h"
#include <libxsmm.h>

at::Tensor mini_dnn::backend::MatmulLibxsmm::forward( at::Tensor i_x,
                                                      at::Tensor i_w ) {
  // get involved sizes
  Matmul::Sizes l_sizes = Matmul::getSizes( i_x,
                                            i_w );

  // create LIBXSMM kernel (isolated)
  libxsmm_gemm_shape l_shape_brgemm;
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;
  
  libxsmm_blasint l_m = l_sizes.bn;
  libxsmm_blasint l_n = l_sizes.bk;
  libxsmm_blasint l_k = l_sizes.bc;

  libxsmm_blasint l_lda = l_m;
  libxsmm_blasint l_ldb = l_k;
  libxsmm_blasint l_ldc = l_m;

  l_shape_brgemm = libxsmm_create_gemm_shape( l_m,
                                              l_n,
                                              l_k,
                                              l_lda,
                                              l_ldb,
                                              l_ldc,
                                              LIBXSMM_DATATYPE_F32,
                                              LIBXSMM_DATATYPE_F32,
                                              LIBXSMM_DATATYPE_F32,
                                              LIBXSMM_DATATYPE_F32 );

  // generate kernel
  // correct values have to be entered
  // Werte müssen hier noch angepasst werden.
  // l_n/l_m/l_lda/l_dc - Test failed (2 of 3)
  // l_k/l_m/l_lda/l_dc - Segmentation fault (core dumped)
  // l_n/l_k/l_lda/l_dc - Test failed (2 of 3)
  // l_m/l_k/l_lda/l_dc - Test failed (2 of 3)

  
  libxsmm_meltw_unary_shape l_unary_shape = libxsmm_create_meltw_unary_shape( l_m,    // m
                                                                              l_n,    // n
                                                                              l_lda,  // lda
                                                                              l_ldc,  // ldc,
                                                                              LIBXSMM_DATATYPE_F32,   // dtype_in
                                                                              LIBXSMM_DATATYPE_F32,   // dtype_out
                                                                              LIBXSMM_DATATYPE_F32 ); // dtype_comp

  
  libxsmm_meltwfunction_unary l_relu = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU,
                                                                        l_unary_shape,
                                                                        LIBXSMM_MELTW_FLAG_UNARY_NONE );

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_xmmfunction l_kernel_forward;
  l_kernel_forward.gemm = libxsmm_dispatch_brgemm_v2( l_shape_brgemm,
                                                      l_flags_brgemm,
                                                      l_prefetch_flags_brgemm,
                                                      l_brconfig );

  libxsmm_gemm_param l_param;
  memset( &l_param,
          0,
          sizeof(libxsmm_gemm_param) );

  // prepare data for blocked LIBXSMM calls
  at::Tensor l_y = at::zeros( {l_sizes.kb, l_sizes.nb, l_sizes.bk, l_sizes.bn} );

  c10::IntArrayRef l_strides_a = i_x.strides();
  c10::IntArrayRef l_strides_b = i_w.strides();
  c10::IntArrayRef l_strides_c = l_y.strides();

  float * l_ptr_a = (float*) i_x.data_ptr();
  float * l_ptr_b = (float*) i_w.data_ptr();
  float * l_ptr_c = (float*) l_y.data_ptr();

  // execute blocked GEMMs for
#pragma omp parallel for collapse(2) firstprivate(l_param)
  for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++ ) {
    for( int64_t l_nb = 0; l_nb < l_sizes.nb; l_nb++ ) {
      int64_t l_offset_c =  l_kb * l_strides_c[0];
              l_offset_c += l_nb * l_strides_c[1];

      for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
        int64_t l_offset_a =  l_nb * l_strides_a[0];
                l_offset_a += l_cb * l_strides_a[1];

        int64_t l_offset_b =  l_kb * l_strides_b[0];
                l_offset_b += l_cb * l_strides_b[1];

        l_param.a.primary = l_ptr_a + l_offset_a;
        l_param.b.primary = l_ptr_b + l_offset_b;
        l_param.c.primary = l_ptr_c + l_offset_c;

        l_kernel_forward.gemm( &l_param );
      }
      libxsmm_meltw_unary_param l_param_relu;
      l_param_relu.in.primary  = l_ptr_c + l_offset_c;
      l_param_relu.out.primary = l_ptr_c + l_offset_c;
      l_relu( &l_param_relu );
    }
  }

  return l_y;
}