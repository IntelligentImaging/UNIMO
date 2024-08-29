
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from se3cnn.image.gated_block import GatedBlock

from . import spatial_mean as sm
from . import points_to_xfm_layer as pxfm


class RXFM_Net(nn.Module):

  def __init__(self,
      input_shape,
      output_chans,
      masks_as_input=False
      ):

      super(RXFM_Net, self).__init__()

      if masks_as_input:
        n_in = 2
      else:
        n_in = 1

      chan_config = [
          [16, 16, 4],
          [16, 16, 4],
          [16, 16, 4],
          [16, 16, 4]
      ]
      features = [ [n_in] ] + chan_config + [ [output_chans] ]

      common_block_params = {
          'size': 5,
          'stride': 1,
          'padding': 2,
          'normalization': None,
          'capsule_dropout_p': None,
          'smooth_stride': False,
      }

      block_params = [{'activation': F.relu}] * (len(features)-2) + [{'activation' : F.relu}]

      assert len(block_params) + 1 == len(features)

      blocks = [
          GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
          for i in range(len(block_params))
      ]

      self.sequence = torch.nn.Sequential(
          *blocks
      )

  def forward(self, x):
      x = self.sequence(x)
      return x

  def to(self, *args, **kwargs):
      self = super().to(*args, **kwargs) 
      return self

class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()

    def scaled_dot_product_attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, volume):
        Q = volume
        K = volume
        V = volume

        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)

        return attention_output, attention_weights

def pts_to_xfms( output_A, output_B, image_shape, weights_A=None,weights_B=None):
  #TODO: do this flip in pxfm
  output_A = torch.transpose(output_A,1,2)
  output_B = torch.transpose(output_B,1,2)

  if weights_A is not None and weights_B is not None:
    weights_A = torch.transpose(weights_A,1,2)
    weights_B = torch.transpose(weights_B,1,2)
    xfm_A2B_R, xfm_A2B_t  = pxfm.rigid_transform_3D_PT_weighted(output_A, output_B, image_shape, weights_A, weights_B)
  else:
    xfm_A2B_R, xfm_A2B_t  = pxfm.rigid_transform_3D_PT(output_A, output_B, image_shape)

  divisor = image_shape.detach()
  divisor = torch.unsqueeze(divisor,0)
  divisor = torch.unsqueeze(divisor,-1)
  divisor = divisor / 2.0

  #print("xfm translation shape",xfm_A2B_t.shape)
  #print("xfm translation ",xfm_A2B_t)
  #print("img shape",image_shape)
  xfm_A2B_t = (xfm_A2B_t / divisor)
  #print("xfm translation ",xfm_A2B_t)

  affine_mats = torch.cat( [xfm_A2B_R, xfm_A2B_t], axis=-1 )

  return affine_mats

# Create positional encoding for the time dimension


##
## this wrapper is to fit the general_runner experimental shape
##
class RXFM_Net_Wrapper(nn.Module):
  def __init__(self,
    input_shape,
    output_chans, 
    masks_as_input=False,
    ):

    super(RXFM_Net_Wrapper, self).__init__()
    self.rxfm_net_obj = RXFM_Net(input_shape, output_chans, masks_as_input)
    self.sm_obj = sm.SpatialMean_CHAN([output_chans] + input_shape, return_power=True, d_model = 32*3)
    self.img_shape_tensor = torch.tensor(input_shape)
    self.att_obj = SelfAttentionLayer()

  def forward( self, x ):
    input_1, input_2 = x
    output_1 = self.rxfm_net_obj.forward(input_1)
    output_2 = self.rxfm_net_obj.forward(input_2)


    means_1, weights_1 = self.sm_obj.forward( output_1 )
    means_2, weights_2 = self.sm_obj.forward( output_2 )

    # print (means_1.shape)
 
    w = weights_1 * weights_2

    xfms = pts_to_xfms(
      means_2, means_1, self.img_shape_tensor, w, w
    )

    return xfms

  def get_channel_outputs( self, x ):
    return self.rxfm_net_obj.forward(x)

  def get_mean_points( self, x ):
    output_1 = self.rxfm_net_obj.forward(x)
    means_1, _ = self.sm_obj.forward( output_1 )
    return means_1

  def to(self,*args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.rxfm_net_obj = self.rxfm_net_obj.to(*args, **kwargs)
    self.sm_obj = self.sm_obj.to(*args, **kwargs)
    self.img_shape_tensor = self.img_shape_tensor.to(*args, **kwargs)
    return self


