#include "Conv2dAtenBlocked.h"

at::Tensor mini_dnn::backend::Conv2dAtenBlocked::forward( at::Tensor i_input,
                                                          at::Tensor i_weight ) {


 // get involved sizes
  Conv2d::Sizes l_sizes = Conv2d::getSizes( i_input,
                                            i_weight );

  // prepare data for blocked Aten calls
  // blocking
  // X: n x cb x h x w x bc
  // W: kb x cb x r x s x bc x bk
  // Y: n x kb x p x q x bk

  at::Tensor l_output = at::zeros( {l_sizes.n, l_sizes.kb, l_sizes.p, l_sizes.q, l_sizes.bk} );

  for( int64_t l_n = 0; l_n < l_sizes.n; l_n++ ) {
    for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++ ) {
      for( int64_t l_p = 0; l_p < l_sizes.p; l_p++ ) {
        for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
          for( int64_t l_r = 0; l_r < l_sizes.r; l_r++ ) {
            for( int64_t l_s = 0; l_s < l_sizes.s; l_s++ ) {
              l_output[l_n][l_kb][l_p] += at::matmul( i_weight[l_kb][l_cb][l_r][l_s],i_input[l_n][l_cb][l_p] );
            }
          }
        }
      }
    }
  }

  return l_output;
}