#include "Conv2dLibxsmm.h"
#include <libxsmm.h>

at::Tensor mini_dnn::backend::Conv2dLibxsmm::forward( at::Tensor i_input,
                                                          at::Tensor i_weight ) {


 // get involved sizes
  Conv2d::Sizes l_sizes = Conv2d::getSizes( i_input,
                                            i_weight );

  // create LIBXSMM kernel (isolated)
  libxsmm_gemm_shape l_shape_brgemm;
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;

  // Block: (bk x bc)(bc x q)
  libxsmm_blasint l_k = l_sizes.bk;
  libxsmm_blasint l_q = l_sizes.q;
  libxsmm_blasint l_c = l_sizes.bc;
  
  libxsmm_blasint l_lda = l_k;
  libxsmm_blasint l_ldb = l_q;
  libxsmm_blasint l_ldc = l_c;

  l_shape_brgemm = libxsmm_create_gemm_shape( l_k,
                                              l_q,
                                              l_c,
                                              l_lda,
                                              l_ldb,
                                              l_ldc,
                                              LIBXSMM_DATATYPE_F32,
                                              LIBXSMM_DATATYPE_F32,
                                              LIBXSMM_DATATYPE_F32,
                                              LIBXSMM_DATATYPE_F32 );


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


  at::Tensor l_output = at::zeros( {l_sizes.n, l_sizes.kb, l_sizes.p, l_sizes.q, l_sizes.bk} );

  c10::IntArrayRef l_strides_a = i_input.strides();
  c10::IntArrayRef l_strides_b = i_weight.strides();
  c10::IntArrayRef l_strides_c = l_output.strides();

  float * l_ptr_a = (float*) i_input.data_ptr();
  float * l_ptr_b = (float*) i_weight.data_ptr();
  float * l_ptr_c = (float*) l_output.data_ptr();



  // prepare data for blocked Aten calls
  // blocking
  // X: n x cb x h x w x bc
  // W: kb x cb x r x s x bc x bk
  // Y: n x kb x p x q x bk
#pragma omp parallel for firstprivate(l_param)
  for( int64_t l_n = 0; l_n < l_sizes.n; l_n++ ) {
    for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++ ) {
      for( int64_t l_p = 0; l_p < l_sizes.p; l_p++ ) {

        int64_t l_offset_c =  l_n * l_strides_c[0];
                l_offset_c += l_kb * l_strides_c[1];
                l_offset_c += l_p * l_strides_c[2];

        for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
          for( int64_t l_r = 0; l_r < l_sizes.r; l_r++ ) {
            for( int64_t l_s = 0; l_s < l_sizes.s; l_s++ ) {
                int64_t l_offset_a =  l_n * l_strides_a[0];
                        l_offset_a += l_cb * l_strides_a[1];
                        l_offset_a += l_p * l_strides_a[2];
                
                int64_t l_offset_b =  l_kb * l_strides_b[0];
                        l_offset_b += l_cb * l_strides_b[1];
                        l_offset_b += l_r * l_strides_b[2];
                        l_offset_b += l_s * l_strides_b[3];
                        
                l_param.a.primary = l_ptr_a + l_offset_a;
                l_param.b.primary = l_ptr_b + l_offset_b;
                l_param.c.primary = l_ptr_c + l_offset_c;

                l_kernel_forward.gemm( &l_param );            
            }
          }
        }
      }
    }
  }

  return l_output;
}