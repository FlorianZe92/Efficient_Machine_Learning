import os
import torch
import torch.distributed

# initialize MPI
torch.distributed.init_process_group( "mpi" )
l_rank = torch.distributed.get_rank()
l_size = torch.distributed.get_world_size()

print( "I am Rank ", l_rank, "of", l_size, "ranks")


#################
# MPI send/recv #
#################

if (l_rank == 0):
    l_data = torch.ones(3,4)
else:
    l_data = torch.zeros(3,4)

if (l_rank ==1):
    print("before send/recv")
    print( l_data )

if (l_rank == 0):
    torch.distributed.send(tensor = l_data,
                           dst    = 1)
elif(l_rank ==1):
    torch.distributed.recv(tensor = l_data,
                           src     = 0)

if (l_rank ==1):
    print("after send/recv")
    print( l_data )



# Reset Values
if (l_rank == 0):
    l_data = torch.ones(3,4)
else:
    l_data = torch.zeros(3,4)

###################
# MPI isend/irecv #
###################

if (l_rank ==1):
    print("before isend/irecv")
    print( l_data )

if (l_rank == 0):
    l_req = torch.distributed.isend(tensor = l_data,
                                   dst    = 1)
elif(l_rank ==1):
    l_req = torch.distributed.irecv(tensor = l_data,
                                   src     = 0)
    print( "before wait" )
    print( l_data )

if (l_rank < 2 ):
    l_req.wait()


if (l_rank ==1):
    print("after wait")
    print( l_data )



#################
# MPI allreduce #
#################

if (l_rank == 0):
    l_data = torch.tensor([ [1,2,3,4],
                            [5,6,7,8],
                            [9,10,11,12],
                            [13,14,15,16] ])
else:
    l_data = torch.tensor([ [1,2,3,4],
                            [5,6,7,8],
                            [9,10,11,12],
                            [13,14,15,16] ])

torch.distributed.all_reduce( l_data,
                              torch.distributed.ReduceOp.SUM)

print( "rank: ", l_rank, "data: ", l_data)
