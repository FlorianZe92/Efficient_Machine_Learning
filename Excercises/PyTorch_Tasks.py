import torch
import numpy as np

shape = (2,3,)

###############################
# Tensor Generating Functions #
###############################

x_ones = torch.ones(shape)
print(f"Ones Tensor: \n {x_ones} \n")
x_zeros = torch.zeros(shape)
print(f"Zeros Tensor: \n {x_zeros} \n")
x_rand = torch.rand(shape)
print(f"Rand Tensor: \n {x_rand} \n")
x_ones_like = torch.ones_like(x_zeros)
print(f"Ones Like Tensor: \n {x_ones_like} \n")

##########################
# List of Lists of Lists #
##########################

data = [[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]],[[12,13,14],[15,16,17]],[[18,19,20],[21,22,23]]]
tensor_lists = torch.tensor(data)
print(f"Lists Tensor: \n {tensor_lists} \n")

####################################
# Using Numpy Array for Converting #
####################################
array_lists = np.array(data)
tensor_array = torch.from_numpy(array_lists)

print(f"Lists Array: \n {array_lists} \n")
print(f"Tensor out of Array: \n {tensor_array} \n")


##############
# Operations #
##############
data_p = [[0,1,2],[3,4,5]]
data_q = [[6,7,8],[9,10,11]]

tensor_p = torch.tensor(data_p)
tensor_q = torch.tensor(data_q)

# Elementwise Operations
torch.add(tensor_p,1)
tensor_p + 1

torch.mul(tensor_q,2)
tensor_q*2

# Matrix-Matrix Product
tensor_q_transpose = torch.transpose(tensor_q,0,1)
mul_01 = tensor_p.matmul(tensor_q_transpose)
mul_02 = tensor_p @ tensor_q_transpose

# Reduction Functions
torch.sum(tensor_p)
torch.max(tensor_p)

# Difference of Code Snippets

# In this Example also l_tensor_0 will be changed to a null matrix
l_tmp = l_tensor_0
l_tmp[:] = 0

# In this Example the Tensor l_tensor_1 will be cloned and the change in the second line does not belong to l_tensor_1
l_tmp = l_tensor_1.clone().detach()
l_tmp[:] = 0


###########
# Storage #
###########

#!# Layout ?

tensor = torch.rand_like(tensor_lists, dtype = torch.float32)
print(f"Shape of tensor: {tensor.shape}")
# Stride is the jump necessary to go from one element to the next one in the specified dimension
print(f"Stride of tensor: {tensor.stride()}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

l_tensor_float = torch.rand_like(tensor,dtype = torch.float32)

# Fix of Second Dimension (Describe Changes)
l_tensor_fixed = l_tensor_float[:,0,:]

# Changes:
# shape from torch.Size([4, 2, 3]) to torch.Size([4, 3])
# stride from (6, 3, 1) to (6,1)

# No Changes:
# dtype + layout + device

# Complex change
l_tensor_complex_view = tensor[::2,1,:]

# Not Sure: ::2 means Module 2 and those Lines are being used in the first Dimension
# shape from torch.Size([4, 2, 3]) to torch.Size([2, 3])
# stride from (6, 3, 1) to (12,1) - German: Hängt von Lage im Speicher ab (Unklar, warum die andere Dimension dann nicht 2 ist, da ja eine weitere Liste "im Weg ist"

# contiguoas function (Alligned in Memory)

l_tensor_contiguous = l_tensor_complex_view.contiguous()
# stride from (12,1) to (3,1)

# Last Task: Did not work on my Computer (TypeError: Integer Expected)
# l_data_raw = (ctypes.c_float).from_address(new) (New was a Tensor)


