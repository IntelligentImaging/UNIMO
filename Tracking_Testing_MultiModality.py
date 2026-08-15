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


LAMBDA_RAW = 1.00
LAMBDA_MAX = 20.0
lambda_eff = min(max(LAMBDA_RAW / LAMBDA_MAX, 0.0), 1.0)

VOXEL_MM = 3.0

epislon = 1e-5

checkpoint = torch.load('./saved_model/best_model_all_modality_multi_obj12.pth')
json_file_val = './test.json'
dataset_val = SData(json_file_val, "test")

'''Set device (GPU or CPU)'''
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading data on:", dev)
'''Create a DataLoader'''
valloader = DataLoader(dataset_val, batch_size=1, shuffle=False)

IMG_SIZE = [96, 96, 96]
n_conv_chan = 1
n_chan = 64
overfit = True
voxel_bound = 5
trans_arr = np.zeros(len(valloader))
angular_arr = np.zeros(len(valloader))

directory_path = "./check_result"
if not os.path.exists(directory_path):
    os.mkdir(directory_path)
net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)
net_obj.load_state_dict(checkpoint['unimo_state_dict'])
net_obj = net_obj.to(dev)
net_obj.eval()

lambda_t = torch.tensor(lambda_eff, device=dev, dtype=torch.float32)


def rotation_geodesic_deg(R_est, R_gt):
    """Geodesic distance on SO(3), in degrees."""
    R_rel = R_est.transpose(-1, -2) @ R_gt
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    return torch.rad2deg(torch.acos(torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)))


'''Testing'''
with torch.no_grad():
    for idx, image_data in enumerate(valloader):
        source, tag = image_data
        ''' Correct the contrast when images contain multiple objects, e.g., LungCT'''
        if (tag[0] == "Multi"):
            source = 1 - source
        b = source.shape[0]
        source = source.to(dev).float()
        rx_train = random.uniform(-math.pi, math.pi)
        ry_train = random.uniform(-math.pi, math.pi)
        rz_train = random.uniform(-math.pi, math.pi)

        # For tx, ty, tz (translation values from -5 to 5)
        tx_train = random.uniform(-voxel_bound, voxel_bound)
        ty_train = random.uniform(-voxel_bound, voxel_bound)
        tz_train = random.uniform(-voxel_bound, voxel_bound)

        mat = ci3d.create_transform(
            rx=rx_train, ry=ry_train, rz=rz_train,
            tx=2.0 * tx_train / IMG_SIZE[0],
            ty=2.0 * ty_train / IMG_SIZE[1],
            tz=2.0 * tz_train / IMG_SIZE[2]
        )
        mat = mat[np.newaxis, :, :]
        mat = mat[:, 0:3, :]
        mat = torch.tensor(mat).float()
        grids = torch.nn.functional.affine_grid(mat, [1, 1] + IMG_SIZE).to(dev)
        target = torch.nn.functional.grid_sample(source, grids, mode="bilinear",
                                                 padding_mode='border', align_corners=True)

        # Ground-truth transform used to synthesise the target
        mat_gt = mat.to(dev)
        R_gt = mat_gt[:, :, 0:3]
        T_gt = mat_gt[:, :, 3]

        # Convert to binary labels
        src_lb = (source >= epislon).to(dev).float()
        tar_lb = (target >= epislon).to(dev).float()

        ''' Compute rigid transformation on both images and shapes '''
        xfm_1to2_I = net_obj.forward((source, target))
        xfm_1to2_G = net_obj.forward((src_lb, tar_lb))

        ''' Fuse two rigid transformations '''
        Q_combined = SI.combine_rigid_transformations(xfm_1to2_I, xfm_1to2_G, lambda_t)

        predicted_grids_a = torch.nn.functional.affine_grid(Q_combined, [1, 1] + IMG_SIZE)
        x_aligned = F.grid_sample(source,
                                  grid=predicted_grids_a,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=True)
        x_aligned_lb = F.grid_sample(src_lb,
                                     grid=predicted_grids_a,
                                     mode='bilinear',
                                     padding_mode='border',
                                     align_corners=True)

        # ------------------------------------------------------------------
        R_est = Q_combined[:, :, 0:3]
        T_est = Q_combined[:, :, 3]
        angular_arr[idx] = float(rotation_geodesic_deg(R_est, R_gt).mean())
        scale = torch.tensor([IMG_SIZE[0] / 2.0, IMG_SIZE[1] / 2.0, IMG_SIZE[2] / 2.0],
                             device=dev, dtype=torch.float32)
        trans_arr[idx] = float(torch.norm((T_est - T_gt) * scale * VOXEL_MM, dim=-1).mean())

        saved = sitk.GetImageFromArray(np.array(source[0, 0, :, :, :].detach().cpu()))
        sitk.WriteImage(saved, './check_result/source_' + str(idx) + '.nii.gz')

        saved = sitk.GetImageFromArray(np.array(src_lb[0, 0, :, :, :].detach().cpu()))
        sitk.WriteImage(saved, './check_result/src_lbl_' + str(idx) + '.nii.gz')

        saved = sitk.GetImageFromArray(np.array(target[0, 0, :, :, :].detach().cpu()))
        sitk.WriteImage(saved, './check_result/target_' + str(idx) + '.nii.gz')

        saved = sitk.GetImageFromArray(np.array(tar_lb[0, 0, :, :, :].detach().cpu()))
        sitk.WriteImage(saved, './check_result/tar_lbl_' + str(idx) + '.nii.gz')

        saved = sitk.GetImageFromArray(np.array(x_aligned[0, 0, :, :, :].detach().cpu()))
        sitk.WriteImage(saved, './check_result/rigid_' + str(idx) + '.nii.gz')

print("translational error (mm) : {:.4f} +/- {:.4f}".format(
    trans_arr.mean(), trans_arr.std(ddof=1)))
print("angular error (deg)      : {:.4f} +/- {:.4f}".format(
    angular_arr.mean(), angular_arr.std(ddof=1)))
