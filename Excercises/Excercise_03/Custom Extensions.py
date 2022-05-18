import torch

# Input Values
l_x = torch.tensor([[7,8,9],[10,11,12]], dtype = torch.float32, requires_grad = True)
l_w = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float32, requires_grad= True)
dl_dy = torch.tensor([[1,2],[2,3]], dtype = torch.float32, requires_grad = True)

# Calculation of Y(X), dl_dx and dl_dw
y_x = torch.matmul(l_x, torch.transpose(l_w, 0, 1))
dl_dx = torch.matmul(dl_dy, l_w)
dl_dw = torch.matmul(torch.transpose(dl_dy, 0, 1),l_x)

# Construction of linear Layer
l_linear_torch = torch.nn.Linear(in_features=3, out_features=2, bias = False)

# Results before changing the weights
l_y = l_linear_torch(l_x)
print("x: ")
print(l_x)
print("y: ")
print(l_y)

# Change manually the weights of the Layer
l_linear_torch.weight = torch.nn.Parameter(l_w)

# Results after changing the weights
l_y = l_linear_torch(l_x)

print("x: ")
print(l_x)
print("y: ")
print(l_y)

# Backward Pass
l_y.retain_grad()
l_y.backward(gradient = dl_dy)

print ("y.grad: ")
print(l_y.grad)
print ("x.grad: ")
print(l_x.grad)
# Not sure why l_w.grad is empty
print("w.grad")
print(l_w.grad)




