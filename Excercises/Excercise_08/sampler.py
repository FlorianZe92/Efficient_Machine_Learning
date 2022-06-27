import torch.utils.data

class SimpleDataSet( torch.utils.data.Dataset ):
  def __init__( self,
                i_length ):
    self.m_length = i_length

  def __len__( self ):
    return self.m_length

  def __getitem__( self,
                   i_idx ):
    return i_idx*10

l_data_simple = SimpleDataSet( 64 )

#########################################################################################
# num_replicas - Bezieht sich auf die Anzahl der beteiligten Prozesse                   #
# drop_last - Abschneiden von Datensatz, wenn dieser nicht exakt aufgeteilt werden kann #
# rank - Betrachteter Prozess                                                           #
# shuffle - Ermöglicht ein shufflen des Datensatzes                                     #
#########################################################################################

l_sampler = torch.utils.data.distributed.DistributedSampler( l_data_simple,
                                                             num_replicas = 2,
                                                             rank = 0,
                                                             shuffle = False,
                                                             drop_last = True)


l_sampler_batch = torch.utils.data.BatchSampler( l_sampler,
                                                 4,
                                                 drop_last = True)


l_data_loader = torch.utils.data.DataLoader( l_data_simple,
                                              batch_sampler = l_sampler_batch)

for l_id, l_x in enumerate( l_data_loader ):
    print( "id: ", l_id, " / x: ", l_x)

