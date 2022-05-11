import torch

## Trains the given MLP-model.
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied (single epoch).
#  @param io_model model which is trained.
#  @param io_optimizer.
#  @return summed loss over all training samples.
def train( i_loss_func,
           io_data_loader,
           io_model,
           io_optimizer ):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  # print(f"Using {device} device")

  # size = len(io_data_loader)
  io_model.train()
  for batch, (X, y) in enumerate(io_data_loader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = io_model(X)
    loss = i_loss_func(pred, y)

    # Backpropagation
    # Sets the gradients of all optimized torch.Tensor s to zero.
    io_optimizer.zero_grad()
    # Back Calculation of Derivates
    loss.backward()
    # Updating the Parameters
    io_optimizer.step()

  return loss.item(), io_model


