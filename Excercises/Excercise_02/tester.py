import torch

## Tests the model
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied.
#  @param io_model model which is tested.
#  @return summed loss over all test samples, number of correctly predicted samples.
def test( i_loss_func,
          io_data_loader,
          io_model):

  device = "cuda" if torch.cuda.is_available() else "cpu"

  l_loss_total = 0
  l_n_correct = 0

  size = len(io_data_loader.dataset)
  num_batches = len(io_data_loader)
  io_model.eval()
  with torch.no_grad():
    for X, y in io_data_loader:
      X, y = X.to(device), y.to(device)
      pred = io_model(X)
      l_loss_total += i_loss_func(pred, y).item()
      l_n_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  l_loss_total /= num_batches
  l_n_correct /= size

  return l_loss_total, l_n_correct


