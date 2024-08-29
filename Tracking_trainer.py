import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import grad
from torch.autograd import Variable
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import sys
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.Diffeo_losses import NCC, MSE, Grad
from src.Diffeo_networks import DiffeoDense  
from SitkDataSet import SitkDataset as SData
from src.tools import ReadFiles as rd
from functools import partial
import src.utils
import src.losses as losses
import src.custom_image3d as ci3d
import src.rxfm_net as rxfm_net
import SimpleITK as sitk 
import time
from pytorch3d import transforms as pt3d_xfms
import math
from src import SphereInterp as SI
'''Read parameters by yaml'''
para = rd.read_yaml('./parameters.yml')

''' Load data by json'''
json_file = './train.json'

batch_size = para.solver.batch_size
dataset = SData(json_file, "train")

'''Set device (GPU or CPU)'''
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading data on:", dev)

'''Create a DataLoader'''
trainloader = DataLoader(dataset, batch_size= batch_size, shuffle=False)

batch_size = para.solver.batch_size
IMG_SIZE = [96,96,96]
loss_func_name = "xfm_6D"

n_conv_chan = 1
n_chan = 64
overfit = True 
running_loss = 0 
epislon = 0.00001
voxel_bound = 5
def_weight = para.solver.def_weight
net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)

if loss_func_name == "xfm_MSE":
    loss_func = partial( losses.xfm_loss_MSE, weight_R=1.0, weight_T=5.0)
elif loss_func_name == "xfm_6D":
    loss_func = partial( losses.xfm_loss_6D, weight_R=1.0, weight_T=5.0)
else:
    print("Loss function not recognized")
    exit(1)
dice_func = partial(losses.dice_loss, hard=False, ign_first_ch=False)
shape_func = partial(losses.curva_loss)
net_obj = net_obj.to(dev)
# Set different learning rates for each network
LR = 0.000025
LR_def = 0.001
optimizer = torch.optim.Adam(net_obj.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0.00001)
criterion = nn.MSELoss()
if (para.model.deformable == True):
    Diffeo_net = DiffeoDense(inshape=(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]),
                      nb_unet_features=[[16, 32,], [32, 32, 16, 16]],
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True,
                      velocity_tag = 0)
    diff_net = Diffeo_net.to(dev)
    optimizer = torch.optim.Adam([
        {'params': net_obj.parameters(), 'lr': LR},
        {'params': diff_net.parameters(), 'lr': LR_def}
    ], lr=0.01) 
    

# '''Training and validation'''      
directory_path = "./check_result"
model_path = "./saved_model"
if not os.path.exists(directory_path):
    os.mkdir(directory_path)  
if not os.path.exists(model_path):
    os.mkdir(model_path)  
auto_weight = torch.tensor(0.0, requires_grad=True, device= dev)
opt= optim.Adam([auto_weight], lr = 0.001) 
for epoch in range(para.solver.epochs):
    total= 0; 

    print('epoch:', epoch)
    for idx, image_data in enumerate(trainloader):
        source, tag=image_data
        ''' Correct the contrast when images contain multiple objects, e.g., LungCT'''
        if (tag[0] == "Multi"):
            source = 1-source

        b = source.shape[0]    
        source = source.to(dev).float() 
        optimizer.zero_grad()   
        rx_train = random.uniform(-math.pi, math.pi)
        ry_train = random.uniform(-math.pi, math.pi)
        rz_train = random.uniform(-math.pi, math.pi)

        # For tx, ty, tz (translation values from -5 to 5)
        tx_train = random.uniform(-voxel_bound, voxel_bound)
        ty_train = random.uniform(-voxel_bound, voxel_bound)
        tz_train = random.uniform(-voxel_bound, voxel_bound)
        #print ("rotation x:", rx_train, "rotation y:", ry_train, "rotation z:", rz_train, "translation x:", tx_train, "translation y:", ty_train, "translation z:", tz_train)
        mat = ci3d.create_transform(
            rx=rx_train, ry=ry_train, rz=rz_train,
            tx=2.0*tx_train/IMG_SIZE[0], ty=2.0*ty_train/IMG_SIZE[1], tz=2.0*tz_train/IMG_SIZE[2]
        )
        mat = mat[np.newaxis,:,:]
        mat = mat[:,0:3,:]
        mat = torch.tensor(mat).float()
        grids = torch.nn.functional.affine_grid(mat, [1,1] + IMG_SIZE).to(dev)
        target = torch.nn.functional.grid_sample(source, grids, mode="bilinear",padding_mode='border',align_corners=True)

        # Convert to binary labels
        src_lb = (source >= epislon).to(dev).float()
        tar_lb = (target >= epislon).to(dev).float()
        # src_lb = gaussian_smoothing((source >= epislon).to(dev).float(), kernel_size = 3, sigma = 0.001) 
        # tar_lb = gaussian_smoothing((target >= epislon).to(dev).float(), kernel_size = 3, sigma = 0.001) 
        saved= sitk.GetImageFromArray(np.array(src_lb[0,0,:,:,:].detach().cpu()))
        save_name = './check_result/src_lbl_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)
        print (src_lb.shape)
        ''' Compute rigid transformation on both images and shapes '''
        xfm_1to2_I = net_obj.forward((source,target))
        xfm_1to2_G = net_obj.forward((src_lb, tar_lb))
        print (xfm_1to2_I.shape)
        # xfm_1to2 = (xfm_1to2_a + auto_weight*xfm_1to2_b)/2
        
        # lambda_param = torch.tensor(auto_weight, device=dev)
        ''' Fuse two rigid transformations '''
        Q_combined = SI.combine_rigid_transformations(xfm_1to2_I, xfm_1to2_G, auto_weight)
        print (Q_combined.shape)

        predicted_grids_a = torch.nn.functional.affine_grid(Q_combined, [1,1] + IMG_SIZE)
        x_aligned = F.grid_sample(source,
                                  grid=predicted_grids_a,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=True)
        predicted_grids_b = torch.nn.functional.affine_grid(Q_combined, [1,1] + IMG_SIZE)
        x_aligned_lb = F.grid_sample(src_lb,
                                  grid=predicted_grids_a,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=True)
        
        if (para.model.deformable == True):
            loss_image = NCC().loss(target, x_aligned) #criterion (target, x_aligned) #NCC().loss(target, x_aligned) #criterion (target, x_aligned)# #rmi_loss(x_aligned, target) 
            disp, deformed = diff_net(x_aligned, target,  registration = True)
            Reg = Grad( penalty= 'l2')
            loss_reg = Reg.loss(disp)
            loss_dist_deform = NCC().loss(target, deformed)
            loss_val = loss_image + .5*loss_dist_deform + .5*loss_reg #+ shape_val + dice_val #loss_val + + def_weight*loss_dist_deform + def_weight*loss_reg
        else: 
            image_loss = NCC().loss(target, x_aligned) + criterion (tar_lb, x_aligned_lb)
            loss_val = image_loss 
        if (abs(loss_val) > 10e-5):
            loss_val.backward(retain_graph=True)
            optimizer.step()
            scheduler.step() 
            with torch.no_grad():
                opt.step()
                opt.zero_grad()
        print ("weight:", auto_weight.item())

        saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
        save_name = './check_result/source_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)

        saved= sitk.GetImageFromArray(np.array(src_lb[0,0,:,:,:].detach().cpu()))
        save_name = './check_result/src_lbl_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)

        saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
        save_name = './check_result/target_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)

        saved= sitk.GetImageFromArray(np.array(tar_lb[0,0,:,:,:].detach().cpu()))
        save_name = './check_result/tar_lbl_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)

        saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
        save_name = './check_result/rigid_' + str(epoch) + '_'+ str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)
        print ("batch loss", loss_val.item())
    print ("training loss:", total)  
    """ @@@@@@@@@@@@@@ Save the trained model@@@@@@@@@@@@@@ """
    checkpoint = {
            'epoch': epoch,
            'unimo_state_dict': net_obj.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
    model_name = "./saved_model/best_model_all_modality_multi_obj" + str(epoch) +".pth"
    torch.save(checkpoint, model_name)