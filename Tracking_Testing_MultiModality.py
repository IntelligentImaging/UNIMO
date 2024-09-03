import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random 
import numpy as np
import os
import src.custom_image3d as ci3d
from SitkDataSet import SitkDataset as SData
import SimpleITK as sitk
import src.rxfm_net as rxfm_net
from pytorch3d import transforms as pt3d_xfms
import time
import math 
from src import SphereInterp as SI
# Load the saved model checkpoint
checkpoint = torch.load('./saved_model/best_model_all_modality_multi_obj12.pth')
json_file_val = './test.json'
dataset_val = SData(json_file_val, "test")

'''Set device (GPU or CPU)'''
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading data on:", dev)
'''Create a DataLoader'''
valloader = DataLoader(dataset_val, batch_size= 1, shuffle=False)

IMG_SIZE = [96,96,96]
n_conv_chan = 1
n_chan = 64
overfit = True 
voxel_bound = 5 
epislon = 10e-3
trans_arr = np.zeros(len(valloader))
angular_arr =  np.zeros(len(valloader))

directory_path = "./check_result"
if not os.path.exists(directory_path):
    os.mkdir(directory_path)  
net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)
net_obj.load_state_dict(checkpoint['unimo_state_dict'])
net_obj = net_obj.to(dev)
net_obj.eval()

'''Testing'''        

for idx, image_data in enumerate(valloader):
    source, tag = image_data
    ''' Correct the contrast when images contain multiple objects, e.g., LungCT'''
    if (tag[0] == "Multi"):
        source = 1-source
    b = source.shape[0]    
    source = source.to(dev).float()   
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
    save_name = './check_result/src_lbl_'  + '_'+ str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
    print (src_lb.shape)
    ''' Compute rigid transformation on both images and shapes '''
    xfm_1to2_I = net_obj.forward((source,target))
    xfm_1to2_G = net_obj.forward((src_lb, tar_lb))
    print (xfm_1to2_I.shape)
    ''' Fuse two rigid transformations '''
    Q_combined = SI.combine_rigid_transformations(xfm_1to2_I, xfm_1to2_G, 0.0)
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

    saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
    save_name = './check_result/source_' + str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(src_lb[0,0,:,:,:].detach().cpu()))
    save_name = './check_result/src_lbl_' + str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
    save_name = './check_result/target_' + str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(tar_lb[0,0,:,:,:].detach().cpu()))
    save_name = './check_result/tar_lbl_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)

    saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
    save_name = './check_result/rigid_' +  str(idx) + '.nii.gz'
    sitk.WriteImage(saved, save_name)
