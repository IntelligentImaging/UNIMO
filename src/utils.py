

import torch
from torch.nn import functional as F
import nibabel as nib
import numpy as np


# NCWHD


def crop_to_size( vol, size ):

  IMG_SIZE = list(vol.size())
  slices = [slice(0,IMG_SIZE[0]),slice(0,IMG_SIZE[1])] #batch and channel
  IMG_SIZE = IMG_SIZE[2:]

  for idx in range(3):
    if IMG_SIZE[idx] < size[idx]:
      slices.append(slice(0,IMG_SIZE[idx]))
      continue

    diff = (IMG_SIZE[idx] - size[idx]) // 2
    mod = (IMG_SIZE[idx] - size[idx]) % 2
    slices.append(slice(diff, IMG_SIZE[idx] - diff - mod) )

  vol = vol[tuple(slices)]

  return vol

def pad_to_size( vol, size ):

  pads = []
  for i in range(3):
    raw_diff = size[-1] - vol.size()[-1]
    diff = raw_diff // 2
    mod = raw_diff % 2
    if raw_diff < 0:
      pads.extend([0,0])
    else:
      pads.extend([diff, diff + mod])

  return F.pad( vol, pads, mode="constant", value=0 )

# NCWHD
def unpad( vol, original_size):
  return vol[:,:,0:original_size[0],0:original_size[1],0:original_size[2]]

def quantile_normalizer(vol, q1=0.90, q2=0.99):
  vol = vol.double()
  q = torch.quantile(vol,torch.tensor([q1,q2],dtype=torch.float64))
  vol = torch.clip(vol,min=0,max=q[1]) / q[0]
  return vol

def load_scale_and_pad( vol_path, size, initial_resize=[128,128,128],rescale=[96,96,96]):
  img = nib.load(vol_path)
  aff = img.affine
  vol = img.get_fdata()
  original_size = vol.shape

  vol = vol[np.newaxis,np.newaxis,:,:,:]
  vol = torch.tensor(vol)
  vol = quantile_normalizer(vol)

  vol = pad_to_size(vol, initial_resize)
  vol = F.interpolate( vol, size=rescale, mode="trilinear", align_corners=False )
  vol = pad_to_size(vol, size)

  return vol, original_size, aff

def apply_transform_torch( x, xfm, order=1):

  IMG_SIZE = list(x.size())[2:]

  with torch.no_grad():
    grids = torch.nn.functional.affine_grid(xfm, [1,1] + IMG_SIZE)
    if order > 0:
      mode="bilinear"
    else:
      mode="nearest"

    x = torch.nn.functional.grid_sample(
      x, grids, mode=mode
    ).detach()
  
  return x




