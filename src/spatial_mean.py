

import numpy as np
import torch
from torch import nn

class SpatialMean_CHAN(nn.Module):
  """ 
  Spatial Mean CHAN (mean coordinate of non-neg. weight tensor)
  This computes offset from CENTER of 0th index voxel, assuming last index
  is channel dim.

  INPUT: Tensor [ Batch x Channels x H x W x D ]
  OUTPUT: Tensor [ BATCH x Channels x 3]

  """

  def __init__(self, input_shape, eps=1e-9, pytorch_order=True, return_power=False, d_model = 32*3, **kwargs):

    super(SpatialMean_CHAN, self).__init__(**kwargs)

    self.eps = eps
    self.size_in = input_shape 

    self.coord_idx_list = []
    self.input_shape_nb_nc = input_shape[1:]
    self.n_chan = input_shape[0]
    self.d_model = d_model
    for idx,in_shape in enumerate(self.input_shape_nb_nc):
      coord_idx_tensor = torch.range(0,in_shape-1)
      coord_idx_tensor = torch.reshape(
        coord_idx_tensor,
        [in_shape] + [1]*(len(self.input_shape_nb_nc)-1)
      )

      #coord_idx_tensor = torch.repeat_interleave(
      #  coord_idx_tensor,
      #  torch.tensor([1] + self.input_shape_nb_nc[:idx] + self.input_shape_nb_nc[idx+1:])
      #)
      coord_idx_tensor = coord_idx_tensor.repeat(*([1] + self.input_shape_nb_nc[:idx] + self.input_shape_nb_nc[idx+1:]))

      coord_idx_tensor = coord_idx_tensor.permute(
        *(list(range(1,idx+1)) + [0] + list(range(idx+1,len(self.input_shape_nb_nc))))
      )

      self.coord_idx_list.append(
        torch.reshape(coord_idx_tensor,[-1])
      )
      #self.coord_idx_list.append(
      #  torch.reshape(torch.cast(coord_idx_tensor, tf.float32),[-1])
      #)

    print("WARNING [spatial_mean]: pytorch reverses its axes because why not.",
      "Thus, pytorch(z,y,x) is the order output unless specified via:\n",
      "pytorch_order=False\n",
      "This order is NOT reversed (i.e. left ambiguous) in their affine_grid",
      sep=" ")
    self.pytorch_order = pytorch_order
    if pytorch_order:
      self.coord_idx_list.reverse()

    self.coord_idxs = torch.stack(self.coord_idx_list)
    self.coord_idxs = torch.unsqueeze(self.coord_idxs, 0)
    self.coord_idxs = torch.unsqueeze(self.coord_idxs, 0)
    #TODO Here we can (re)map over the channel dims instead of tilling
    #self.coord_idxs = self.coord_idxs.repeat(self.n_chan,1,1)
    self.angle_rates = (1 / torch.pow(10000, (2 * torch.arange(self.d_model).float() / float(self.d_model)))).to('cuda')
    self.return_power = return_power

  def _apply_coords(self,x, verbose=False):
    #!ALERT This is what the warning is about

    if verbose:
      print(x.shape)
      print(self.coord_idxs.shape)

    #x = torch.unsqueeze( x, 1 ).repeat(1,3,1)
    x = torch.unsqueeze( x, 2 )

    if verbose:
      print(x.shape)

    numerator = torch.sum( x*self.coord_idxs, dim=[3])

    #here, we do not want the gradient to see a normalization.
    denominator = torch.sum(torch.abs(x.detach()) + self.eps,dim=[3])

    if verbose:
      print(numerator.shape)
      print(denominator.shape)
    return numerator / denominator

  def forward(self, x):
    #batch
    #x = keras.backend.batch_flatten(x)
    x = torch.reshape( x, [-1, self.n_chan, np.prod(self.input_shape_nb_nc)] )
    x = torch.abs(x)
    #print("power by chan",power_by_chan)

    #haha these are some axes.
    #but really, they're the axes we sum over
    #sum_axes = list(range(len(self.input_shape_nb_nc)))
    #sum_axes = [1] #list(range(len(self.input_shape_nb_nc)))

    #outputs = torch.vmap( self._apply_coords )(x)
    #outputs = []
    outputs = self._apply_coords(x)
    #for idx in range( x.size(0) ):
    #  outputs.append(self._apply_coords(x[idx]))
    #outputs = torch.stack(outputs) 

    if self.return_power:
      power_by_chan = x.sum(dim=2,keepdim=True)
      return outputs, power_by_chan
    return outputs #K.mean(K.batch_flatten(K.square(x[0] - x[1])), -1)

  def time_encoder(self, position):
    positions = position * self.angle_rates
    positions = positions.view(-1, self.d_model // 3)

    pos_encoding = torch.cat([torch.sin(positions), torch.cos(positions)], dim=-1)

    return pos_encoding.permute(1,0).unsqueeze(0)


  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.coord_idxs = self.coord_idxs.to(*args, **kwargs) 
    for idx in range(len(self.coord_idx_list)):
      self.coord_idx_list[idx].to(*args, **kwargs)
    return self

