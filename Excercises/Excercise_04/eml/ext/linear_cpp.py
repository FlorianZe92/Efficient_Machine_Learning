import torch.autograd
import torch.nn
import eml_ext_linear_cpp


class Function( torch.autograd.Function ):
    @staticmethod
    def forward(io_ctx, i_input, i_weights):
         # Kontext wird dafür genutzt, dass man beim Forward Pass schon Daten angefasst/erstellt hat, welche dann im Backward Pass benötigt werden
        io_ctx.save_for_backward(i_input, i_weights)
        l_input = i_input.contiguous()
        l_weights = i_weights.contiguous()
        l_output  = eml_ext_linear_cpp.forward(l_input,l_weights)
        return l_output

    @staticmethod
    def backward(io_ctx, i_grad_output):
        #i_grad_output entspricht dL/dy
        #print("Hello from backward Start")
        i_input, i_weights = io_ctx.saved_tensors
        l_input = i_input.contiguous()
        l_weights = i_weights.contiguous()
        l_vector = eml_ext_linear_cpp.backward(i_grad_output, l_input, l_weights)
        l_grad_input = l_vector[0]
        l_grad_weights = l_vector[1]
        print("l_grad_input")
        print(l_grad_input)
        print("l_grad_weights")
        print(l_grad_weights)
        return l_grad_input, l_grad_weights
        #return l_vector

## @package eml.ext.linear_python.Layer
#  Custom implementation of a linear layer in python.
class Layer( torch.nn.Module ):
  ## Initialize the linear layer.
  #  @param i_n_features_input number of input features.
  #  @param i_n_features_output number of output features.
  def __init__( self,
                i_n_features_input,
                i_n_features_output ):
    super( Layer,
           self ).__init__()

    # store weight matrix A^T
    self.m_weights = torch.nn.Parameter( torch.Tensor( i_n_features_input,
                                                       i_n_features_output ) )

  ## Forward pass of the linear layer.
  #  @param i_input input data.
  #  @return input @ weights.
  def forward( self,i_input ):
    return Function.apply(i_input, self.m_weights)

