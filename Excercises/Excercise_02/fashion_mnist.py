import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from torch.utils.tensorboard import SummaryWriter

## Converts an Fashion MNIST numeric id to a string.
#  @param i_id numeric value of the label.
#  @return string corresponding to the id.
def toLabel( i_id ):
  l_labels = [ "T-Shirt",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Shirt",
               "Sneaker",
               "Bag",
               "Ankle Boot" ]

  return l_labels[i_id]

## Applies the model to the data and plots the data.
#  @param i_off offset of the first image.
#  @param i_stride stride between the images.
#  @param io_data_loader data loader from which the data is retrieved.
#  @param io_model model which is used for the predictions.
#  @param i_path_to_pdf optional path to an output file, i.e., nothing is shown at runtime.
def plot( i_off,
          i_stride,
          io_data_loader,
          io_model, epoche,
          i_path_to_pdf = None ):

  device = "cuda" if torch.cuda.is_available() else "cpu"

  io_model = io_model.to(device)

  # switch to evaluation mode
  io_model.eval()

  pic_amount = 50
  pic_counter = 0
  # create pdf if required
  if( i_path_to_pdf == None ):
      name = "Visualization_after_" + str(epoche) + "_Epochs.pdf"
      with PdfPages(name) as pdf:
         for X, y in io_data_loader:
             if pic_counter >= pic_amount:
                 break
             Data_X = X
             Data_X = Data_X.to(device)
             pred = io_model(Data_X)
             # print(pred.argmax(1))
             for i in range(len(X)):
                 if (i_off - (i % i_stride)) == 0:
                    pic_counter+=1
                    plt.figure
                    plt.imshow(X[i][0,:,:], cmap="gray")
                    plt.title(f'Label: {toLabel(y[i])} Prediction: {toLabel(pred.argmax(1)[i])}')
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()



