#!/usr/bin/python3
import torch
import torchvision.datasets
import torchvision.transforms
import torch.utils.data

import eml.mlp.model
import eml.mlp.trainer
import eml.mlp.tester

import time

# import eml.vis.fashion_mnist

# init MPI
torch.distributed.init_process_group( "mpi" )
l_rank = torch.distributed.get_rank()
l_size = torch.distributed.get_world_size()

if (l_rank == 0):
  print( "################################" )
  print( "# Welcome to EML's MLP example #" )
  print( "################################" )


# set up datasets
if (l_rank == 0):
  print( 'setting up datasets')
l_data_train = torchvision.datasets.FashionMNIST( root      = "data/fashion_mnist",
                                                  train     = True,
                                                  download  = True,
                                                  transform = torchvision.transforms.ToTensor() )

l_data_test = torchvision.datasets.FashionMNIST( root      = "data/fashion_mnist",
                                                 train     = False,
                                                 download  = True,
                                                 transform = torchvision.transforms.ToTensor() )


# Distributed Data Sampler

l_sampler_train = torch.utils.data.distributed.DistributedSampler( l_data_train,
                                                                   num_replicas = l_size,
                                                                   rank = l_rank,
                                                                   shuffle = False,
                                                                   drop_last = True)

l_sampler_test = torch.utils.data.distributed.DistributedSampler(  l_data_test,
                                                                   num_replicas = l_size,
                                                                   rank = l_rank,
                                                                   shuffle = False,
                                                                   drop_last = True)

# Batch Samplers for train+test using micro batch sizes

l_batch_size = 64
l_micro_batch_size = l_batch_size // l_size

l_sampler_batch_train = torch.utils.data.BatchSampler( l_sampler_train,
                                                       l_micro_batch_size,
                                                       drop_last = True)

l_sampler_batch_test = torch.utils.data.BatchSampler(  l_sampler_test,
                                                       l_micro_batch_size,
                                                       drop_last = True)


# init data loaders
if (l_rank == 0):
  print( 'initializing data loaders' )
l_data_loader_train = torch.utils.data.DataLoader( l_data_train,
                                                   batch_sampler = l_sampler_batch_train )
l_data_loader_test  = torch.utils.data.DataLoader( l_data_test,
                                                   batch_sampler = l_sampler_batch_test )

# set up model, loss function and optimizer
if (l_rank == 0):
  print( 'setting up model, loss function and optimizer' )
l_model = eml.mlp.model.Model()
l_loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.SGD( l_model.parameters(),
                               lr = 1E-3 )
if (l_rank == 0):
  print( l_model )

# train for the given number of epochs

if (l_rank == 0):
  start_time = time.time()


l_n_epochs = 20
for l_epoch in range( l_n_epochs ):
  if (l_rank == 0):
    print( 'training epoch #' + str(l_epoch+1) )
  l_loss_train = eml.mlp.trainer.train( l_loss_func,
                                        l_data_loader_train,
                                        l_model,
                                        l_optimizer,
                                        l_size )

  # Schönere Schreibweise hierfür überlegen und den Loss Wert dann direkt ausgeben lassen
  l_loss_train_tensor = torch.full((1, 1), l_loss_train)
  torch.distributed.all_reduce( l_loss_train_tensor,
                                op = torch.distributed.ReduceOp.SUM)
                              
  if (l_rank == 0):                                      
    print( '  training loss:', l_loss_train_tensor.item() )


  # Hier müsste auch Allreduce verwendet werden
  l_loss_test, l_n_correct_test = eml.mlp.tester.test( l_loss_func,
                                                       l_data_loader_test,
                                                       l_model )
  l_loss_test_tensor = torch.full((1, 1), l_loss_test)
  l_n_correct_test_tensor = torch.full((1, 1), l_n_correct_test)
  torch.distributed.all_reduce( l_loss_test_tensor,
                                op = torch.distributed.ReduceOp.SUM)
  torch.distributed.all_reduce( l_n_correct_test_tensor,
                                op = torch.distributed.ReduceOp.SUM)

  if (l_rank == 0):
    l_accuracy_test = l_n_correct_test_tensor / len(l_data_loader_test.dataset)
    print( '  test loss:', l_loss_test_tensor.item() )
    print( '  test accuracy:', l_accuracy_test.item() )

  # visualize results of intermediate model every 10 epochs
#  if( (l_epoch+1) % 10 == 0 ):
#    l_file_name =  'test_dataset_epoch_' + str(l_epoch+1) + '.pdf'
#    print( '  visualizing intermediate model w.r.t. test dataset: ' + l_file_name )
#    eml.vis.fashion_mnist.plot( 0,
#                                250,
#                                l_data_loader_test,
#                                l_model,
#                                l_file_name )

# visualize results of final model
#l_file_name = 'test_dataset_final.pdf'
#print( 'visualizing final model w.r.t. test dataset:', l_file_name )
#eml.vis.fashion_mnist.plot( 0,
 #                           250,
 #                           l_data_loader_test,
 #                           l_model,
 #                           l_file_name )



if (l_rank == 0):
  end_time = time.time()
  time_elapsed = (end_time - start_time)
  print('Gemessene Zeit: ', time_elapsed )


if (l_rank == 0):
  print( "#############" )
  print( "# Finished! #" )
  print( "#############" )