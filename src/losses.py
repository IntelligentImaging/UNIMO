
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk 
#import pytorch3d.transforms as pt3d_xfms

# pool_obj = nn.AvgPool3d( 5, stride=1, padding=2)
def boundary_weighted_loss(true, pred, true_mask, pool_obj, boundary_weight=5):
  err = true - pred 
  pooled = pool_obj.forward( true_mask )
  boundaries = boundary_weight * (pooled > 0).float() * (pooled < 1).float()
  SE = (err * err)
  return (boundaries * SE + SE).mean()

def xfm_loss_MSE( true, pred, weight_R = 1.0, weight_T = 1.0 ):
  true_R = true[:,0:3,0:3]
  print ("true label:", true_R.shape)
  pred_R = pred[:,0:3,0:3]
  print ("prediction", pred.shape)
  err_R = true_R - pred_R
  err_R = (err_R * err_R).mean()

  true_T = true[:,0:3,:]
  pred_T = pred[:,0:3,:]

  err_T = true_T - pred_T
  err_T = (err_T * err_T).mean()

  return weight_R * err_R + weight_T * err_T

#def xfm_loss_6D( true, pred, weight_R = 1.0, weight_T = 1.0 ):
#  true_R = pt3d_xfms.matrix_to_rotation_6d(true[:,0:3,0:3])
#  pred_R = pt3d_xfms.matrix_to_rotation_6d(pred[:,0:3,0:3])
#
#  err_R = true_R - pred_R
#  err_R = (err_R * err_R).mean()
#
#  true_T = true[:,0:3,:]
#  pred_T = pred[:,0:3,:]
#
#  err_T = true_T - pred_T
#  err_T = (err_T * err_T).mean()
#
#  return weight_R * err_R + weight_T * err_T

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    return theta

def xfm_loss_6D( true, pred, weight_R = 1.0, weight_T = 1.0 ):
    true_R = true[:,0:3,0:3]
    pred_R = pred[:,0:3,0:3]

    err_R = compute_geodesic_distance_from_two_matrices(true_R, pred_R)
    err_R = (err_R * err_R).mean()

    true_T = true[:,0:3,:]
    pred_T = pred[:,0:3,:]

    err_T = true_T - pred_T
    err_T = (err_T * err_T).mean()

    return weight_R * err_R + weight_T * err_T


def xfm_loss_geodesic( true, pred, weight_R = 1.0, weight_T = 1.0 ):
  # eq 10 from 
  # https://arxiv.org/abs/1803.05982

  true_R = true[:,0:3,0:3]
  pred_R = pred[:,0:3,0:3]

  true_R = true_R.float()
  pred_R = pred_R.float()

  err_R = (torch.einsum('bii->b',torch.matmul(pred_R,true_R)) - 1) / 2
  err_R = (torch.arccos(err_R)).mean()

  true_T = true[:,0:3,:]
  pred_T = pred[:,0:3,:]

  err_T = true_T - pred_T
  err_T = (err_T * err_T).mean()

  return weight_R * err_R + weight_T * err_T




def dice_loss( pred, target, hard=False, ign_first_ch=False):
    eps = 1
    assert pred.size() == target.size(), 'Input and target are different dim'
    
    if len(target.size())==4:
        n,c,_,_ = target.size()
    if len(target.size())==5:
        n,c,_,_,_ = target.size()

    target = target.contiguous().view(n,c,-1)
    pred = pred.contiguous().view(n,c,-1)
    
    if hard == True: # hard Dice
        pred_onehot = torch.zeros_like(pred)
        pred = torch.argmax(pred, dim=1, keepdim=True)
        pred = torch.scatter(pred_onehot, 1, pred, 1.)
    if ign_first_ch:
        target = target[:,1:,:]
        pred = pred[:,1:,:]
  
    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    ind_avg = dice_loss
    total_avg = torch.mean(dice_loss)
    regions_avg = torch.mean(dice_loss, 0)
    
    if hard == False:
        return total_avg
    else:
        return total_avg, regions_avg, ind_avg






# def compute_curvature(volume):
#     # Define convolution kernels for first and second derivatives
#     kernel_dx = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32)
#     kernel_dy = torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32)
#     kernel_dzz = torch.tensor([[[[1, -2, 1]]]], dtype=torch.float32)
#     print (kernel_dx.shape)
#     # Compute gradients using convolutions
#     gradient_x = F.conv3d(volume, kernel_dx, padding=1, stride=1)
#     gradient_y = F.conv3d(volume, kernel_dy, padding=1, stride=1)
#     gradient_z = F.conv3d(volume, kernel_dzz, padding=1, stride=1)

#     # Compute second-order derivatives using convolutions
#     dxx = F.conv3d(gradient_x, kernel_dx.permute(0, 1, 3, 2), padding=1, stride=1)
#     dyy = F.conv3d(gradient_y, kernel_dy.permute(0, 1, 3, 2), padding=1, stride=1)
#     dzz = F.conv3d(gradient_z, kernel_dzz.permute(0, 1, 3, 2), padding=1, stride=1)

#     # Compute curvature as the mean of second derivatives
   

#     return curvature

def compute_curvature(volume):
    # Compute gradients along each axis
    binary_mask = (volume > 10e-5 ).float()
    # Compute gradients along each axis
    gradients = torch.gradient(binary_mask, dim=(2,3,4))

    # Extract second-order derivatives
    dxx = torch.gradient(gradients[0], dim=(2))[0]
    dyy = torch.gradient(gradients[1], dim=(3))[0]
    dzz = torch.gradient(gradients[2], dim=(4))[0]

    # Compute curvature as the mean of second derivatives
    curvature = (dxx + dyy + dzz) / 3.0
    

    return curvature

def curva_loss(pred, target):
    curvature1 = compute_curvature(pred)

    curvature2 = compute_curvature(target)

    return F.l1_loss(curvature1, curvature2), curvature1, curvature2

def batch_compute_fft(volumes):
    # Compute FFT of the volumes along each dimension
    fft_volumes = torch.fft.fftn(volumes, dim=(-3, -2, -1))

    return fft_volumes

def batch_compute_fft_dissimilarity(volumes1, volumes2):
    # Compute FFT of the volumes
    fft_volumes1 = batch_compute_fft(volumes1)
    fft_volumes2 = batch_compute_fft(volumes2)

    # Compute magnitude spectra of the FFTs
    magnitude_spectrum1 = torch.abs(fft_volumes1)
    magnitude_spectrum2 = torch.abs(fft_volumes2)

    # Reshape magnitude spectra tensors to have only two dimensions
    magnitude_spectrum1_reshaped = magnitude_spectrum1.view(magnitude_spectrum1.shape[0], -1)
    magnitude_spectrum2_reshaped = magnitude_spectrum2.view(magnitude_spectrum2.shape[0], -1)

    # Compute dissimilarity based on magnitude spectra (e.g., Euclidean distance)
    fft_dissimilarity = torch.norm(magnitude_spectrum1_reshaped - magnitude_spectrum2_reshaped, dim=1)

    return fft_dissimilarity


















