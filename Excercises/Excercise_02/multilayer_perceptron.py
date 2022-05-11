from trainer import train
from tester import test
from model import NeuralNetwork
from fashion_mnist import plot

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
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#######################
# Additional Material #
#######################

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# https://pytorch.org/tutorials/

###############
# Data Loader #
###############

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#################
# Visualization #
#################
"""
with PdfPages('visualization.pdf') as pdf:
    for i in range(10):
        plt.figure
        plt.imshow(training_data[i][0][0,:,:], cmap="gray")
        plt.title(f'Label: {training_data[i][1]} ')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
"""

##########################
# Illustration of shapes #
##########################

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


##################
# Training Model #
##################

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)

# Erläuterung von CrossEntropy: Logarithmus von Output wird verwendet (Angabe von durchschnittlichem Loss am Ende)
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
loss_fn = nn.CrossEntropyLoss()

# Gradient Descent Verfahren
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 101

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, new_model = train(loss_fn, train_dataloader, model, optimizer)
    print(f"loss: {loss:>7f}")
    print("------------------------------------------------")
    l_loss_total, l_n_correct = test(loss_fn, test_dataloader, new_model)
    print(f"Test Error: \n Accuracy: {(100 * l_n_correct):>0.1f}%, Avg loss: {l_loss_total:>8f} \n")
    if t % 10 == 0:
        plot(5,200,test_dataloader,new_model,t)
    # test(test_dataloader, model, loss_fn)
print("Done with Training!")

#l_loss_total, l_n_correct = test(loss_fn, test_dataloader, new_model)
#print(f"Test Error: \n Accuracy: {(100 * l_n_correct):>0.1f}%, Avg loss: {l_loss_total:>8f} \n")




