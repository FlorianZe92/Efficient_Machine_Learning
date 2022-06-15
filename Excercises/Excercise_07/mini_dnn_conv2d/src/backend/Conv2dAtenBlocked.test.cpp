#include <catch2/catch.hpp>
#include "Conv2dAtenBlocked.h"

TEST_CASE( "Tests the convolution aten blocked multiplication",
           "[conv2d][atenblocked][forward]" ) {

  int64_t l_size_n = 3;
  int64_t l_size_h = 8;
  int64_t l_size_w = 12;
  int64_t l_size_c = 6;

  int64_t l_size_k = 4;
  int64_t l_size_r = 3;
  int64_t l_size_s = 3;

  //dimensions for y
  int64_t l_size_p = l_size_h - (l_size_r / 2)*2;
  int64_t l_size_q = l_size_w - (l_size_s / 2)*2;

  //add dimensions for blocking
  int64_t l_size_bc =  3;
  int64_t l_size_bk =  2;

  int64_t l_size_kb = l_size_k / l_size_bk;
  int64_t l_size_cb = l_size_c / l_size_bc;


  // construct input tensors
  at::Tensor l_input = at::rand( { l_size_n, l_size_c, l_size_h, l_size_w } );
  at::Tensor l_weight = at::rand( { l_size_k, l_size_c, l_size_r, l_size_s } );

  // blocking
  // X: n x cb x h x w x bc
  // W: kb x cb x r x s x bc x bk
  // Y: n x kb x p x q x bk


  //construct the blocked view of the tensors
  at::Tensor l_input_blocked = l_input.view( { l_size_n, l_size_cb, l_size_bc, l_size_h, l_size_w } );
  l_input_blocked = l_input_blocked.permute( { 0, 1, 3, 4, 2} ).contiguous();

  at::Tensor l_weight_blocked = l_weight.view( { l_size_kb, l_size_bk, l_size_cb, l_size_bc, l_size_r, l_size_s } );
  l_weight_blocked = l_weight_blocked.permute( { 0, 2, 4, 5, 3, 1} ).contiguous();


  // compute solution
  mini_dnn::backend::Conv2dAtenBlocked l_conv2d;

  at::Tensor l_output_blocked = l_conv2d.forward( l_input_blocked,
                                                  l_weight_blocked );
  
  // reverse blocking
  at::Tensor l_output = l_output_blocked.permute( {0, 1, 4, 2, 3} ).contiguous();
  l_output = l_output.view( { l_size_n, l_size_k, l_size_p, l_size_q } );


  // compute reference
  at::Tensor l_reference = at::matmul( l_input, l_weight );

  // check solution
  REQUIRE( at::allclose( l_output, l_reference ) );
}