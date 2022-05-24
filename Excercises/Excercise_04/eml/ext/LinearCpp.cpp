#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor forward(torch::Tensor i_input, 
                      torch::Tensor i_weights){
    std::cout << "hello world from forward" << std::endl;

    torch::Tensor l_result = torch::matmul( i_input, i_weights);

    float* temp_input = (float*)i_input.data_ptr();
    float* temp_weights = (float*)i_weights.data_ptr();
    float temp_mult;

    for (int i = 0; i < 0; temp_input.sizes()[0]; ++i){


    }
auto temp_tensor = torch::randn({ 3, 4 }).to(torch::kInt32);

int* temp_arr = temp_tensor.data<int>();
		
// alternative
//int* temp_arr = (int*)temp_tensor.data_ptr();

for (int y = 0; y < temp_tensor.sizes()[0]; ++y)
{
	for (int x = 0; x < temp_tensor.sizes()[1]; ++x)
	{
	    std::cout << (*temp_arr++) << " ";
	}

	std::cout << std::endl;
}



    return l_result;
}

std::vector< torch::Tensor > backward( torch::Tensor i_grad_output,
                                       torch::Tensor i_input,
                                       torch::Tensor i_weights ){
    std::cout << "hello world from backward" << std::endl;
    torch::Tensor l_grad_input = torch::matmul(i_grad_output, torch::transpose(i_weights,0,1));
    torch::Tensor l_grad_weights = torch::matmul(torch::transpose(i_input,0,1),i_grad_output);
    std::vector< torch::Tensor> vec_tensor {l_grad_input,l_grad_weights};
    return vec_tensor;

}
 

PYBIND11_MODULE( TORCH_EXTENSION_NAME,
                 io_module) {
    io_module.def( "forward",
                   &forward,
                   "Forward pass of our C++ extension");

    io_module.def( "backward",
                   &backward,
                   "Backward pass of our C++ extension");

}
