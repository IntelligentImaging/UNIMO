
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
import csv
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.Diffeo_losses import NCC, MSE, Grad
from src.Diffeo_networks import *
from SitkDataSet import SitkDataset as SData
from src.tools import ReadFiles as rd
from functools import partial
import src.utils
import src.losses as losses
import src.custom_image3d as ci3d
import src.rxfm_net as rxfm_net
import time
from pytorch3d import transforms as pt3d_xfms
import math
from src.uEpdiff import Epdiff
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
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

batch_size = para.solver.batch_size
IMG_SIZE = [96, 96, 96]
loss_func_name = "xfm_6D"

n_conv_chan = 1
n_chan = 64
overfit = True
running_loss = 0
epislon = 0.00001
voxel_bound = 5
def_weight = para.solver.def_weight


LAMBDA_MAX = float(os.environ.get("LAMBDA_MAX", 20.0))
LAMBDA_INIT = float(os.environ.get(
    "LAMBDA_INIT",
    getattr(para.solver, "lambda_init", 3.0)
))
RUN_TAG = os.environ.get("RUN_TAG", "init{:g}".format(LAMBDA_INIT))
SAVE_DEBUG_VOLUMES = os.environ.get("SAVE_DEBUG_VOLUMES", "0") == "1"
SAVE_CHECKPOINT_EVERY = int(os.environ.get("SAVE_CHECKPOINT_EVERY", 50))

print("[lambda] init(raw) = {:.4f}  LAMBDA_MAX = {:.1f}  -> init(eff) = {:.4f}".format(
    LAMBDA_INIT, LAMBDA_MAX, min(max(LAMBDA_INIT / LAMBDA_MAX, 0.0), 1.0)))

net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)

if loss_func_name == "xfm_MSE":
    loss_func = partial(losses.xfm_loss_MSE, weight_R=1.0, weight_T=5.0)
elif loss_func_name == "xfm_6D":
    loss_func = partial(losses.xfm_loss_6D, weight_R=1.0, weight_T=5.0)
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
                             nb_unet_features=[[16, 32, ], [32, 32, 16, 16]],
                             nb_unet_conv_per_level=1,
                             int_steps=7,
                             int_downsize=2,
                             src_feats=1,
                             trg_feats=1,
                             unet_half_res=True,
                             velocity_tag=0)
    diff_net = Diffeo_net.to(dev)
    optimizer = torch.optim.Adam([
        {'params': net_obj.parameters(), 'lr': LR},
        {'params': diff_net.parameters(), 'lr': LR_def}
    ], lr=0.01)

# ----------------------------------------------------------------------------
# Output directories
# ----------------------------------------------------------------------------
directory_path = "./check_result"
model_path = "./saved_model"
log_path = "./lambda_logs"
for p in (directory_path, model_path, log_path):
    if not os.path.exists(p):
        os.mkdir(p)

# ----------------------------------------------------------------------------
# Learnable fusion weight
# ----------------------------------------------------------------------------
auto_weight = torch.tensor(LAMBDA_INIT, requires_grad=True, device=dev)
opt = optim.Adam([auto_weight], lr=0.001)

# Trajectory log for re-plotting Figure 2
traj_file = os.path.join(log_path, "lambda_traj_{}.csv".format(RUN_TAG))
traj_fh = open(traj_file, "w", newline="")
traj_writer = csv.writer(traj_fh)
traj_writer.writerow(["epoch", "iter", "lambda_raw", "lambda_eff", "lambda_grad", "loss"])

grad_checked = False

for epoch in range(para.solver.epochs):
    total = 0

    print('epoch:', epoch)
    for idx, image_data in enumerate(trainloader):
        source, tag = image_data
        ''' Correct the contrast when images contain multiple objects, e.g., LungCT'''
        if (tag[0] == "Multi"):
            source = 1 - source

        b = source.shape[0]
        source = source.to(dev).float()

        # --- zero BOTH optimisers before the forward/backward pass ---
        optimizer.zero_grad()
        opt.zero_grad()

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

        # Convert to binary labels
        src_lb = (source >= epislon).to(dev).float()
        tar_lb = (target >= epislon).to(dev).float()

        ''' Compute rigid transformation on both images and shapes '''
        xfm_1to2_I = net_obj.forward((source, target))
        xfm_1to2_G = net_obj.forward((src_lb, tar_lb))

        lambda_eff = torch.clamp(auto_weight / LAMBDA_MAX, 0.0, 1.0)

        ''' Fuse two rigid transformations (SLERP on SO(3)) '''
        Q_combined = SI.combine_rigid_transformations(xfm_1to2_I, xfm_1to2_G, lambda_eff)

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

        if (para.model.deformable == True):
            loss_image = NCC().loss(target, x_aligned) + criterion(tar_lb, x_aligned_lb)
            phiinv_bch = torch.zeros(b, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3).to(dev)
            reg_save = torch.zeros(b, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3).to(dev)

            if (para.model.shooting == "SVF"):
                disp, deformed, temp = diff_net(x_aligned, target, registration=True)
                disp_lbl, deformed_lbl, temp = diff_net(x_aligned_lb, tar_lb, registration=True)
                Reg = Grad(penalty='l2')
                loss_reg = Reg.loss(disp)
                loss_reg_lbl = Reg.loss(disp_lbl)
                loss_dist_deform = NCC().loss(target, deformed)
                loss_dist_deform_lbl = criterion(tar_lb, deformed_lbl)
                loss_val = (loss_image
                            + .5 * loss_dist_deform + .5 * loss_reg
                            + .5 * loss_dist_deform_lbl + .5 * loss_reg_lbl)

            if (para.model.shooting == "FLDDMM"):
                identity = get_grid2(IMG_SIZE[0], dev).permute([0, 4, 3, 2, 1])
                epd = Epdiff(dev,
                             (para.model.reduced_xDim, para.model.reduced_yDim, para.model.reduced_zDim),
                             (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]),
                             para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

                # ---- image branch ----
                disp, deformed, momentum = diff_net(x_aligned, target, registration=True)
                momentum = momentum.permute(0, 4, 3, 2, 1)
                for b_id in range(b):
                    v_fourier = epd.spatial2fourier(
                        momentum[b_id, ...].reshape(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3))
                    velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(
                        IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3)
                    reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                    num_steps = para.solver.Euler_steps
                    v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)
                    phiinv = displacement.unsqueeze(0) + identity
                    phiinv_bch[b_id, ...] = phiinv
                    reg_save[b_id, ...] = reg_temp

                dfm = Torchinterp(x_aligned, phiinv_bch)
                Dist = criterion(dfm, target)
                Reg_loss = reg_save.sum()

                # ---- label (shape) branch: use the LABEL pair, not the image pair ----
                disp_lbl, deformed_lbl, momentum_lbl = diff_net(x_aligned_lb, tar_lb, registration=True)
                momentum_lbl = momentum_lbl.permute(0, 4, 3, 2, 1)

                phiinv_bch_lbl = torch.zeros(b, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3).to(dev)
                reg_save_lbl = torch.zeros(b, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3).to(dev)
                for b_id in range(b):
                    v_fourier = epd.spatial2fourier(
                        momentum_lbl[b_id, ...].reshape(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3))
                    velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(
                        IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3)
                    reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                    num_steps = para.solver.Euler_steps
                    v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)
                    phiinv_lbl = displacement.unsqueeze(0) + identity
                    phiinv_bch_lbl[b_id, ...] = phiinv_lbl
                    reg_save_lbl[b_id, ...] = reg_temp

                dfm_lbl = Torchinterp(x_aligned_lb, phiinv_bch_lbl)
                Dist_lbl = criterion(dfm_lbl, tar_lb)
                Reg_loss_lbl = reg_save_lbl.sum()

                loss_val = (loss_image
                            + Dist + para.solver.def_weight * Reg_loss
                            + Dist_lbl + para.solver.def_weight * Reg_loss_lbl)
        else:
            image_loss = NCC().loss(target, x_aligned) + criterion(tar_lb, x_aligned_lb)
            loss_val = image_loss


        if (abs(loss_val) > 10e-5):
            loss_val.backward(retain_graph=True)
            optimizer.step()
            opt.step()
            scheduler.step()

        lam_raw = float(auto_weight.detach())
        lam_eff = float(lambda_eff.detach())
        lam_grad = float(auto_weight.grad) if auto_weight.grad is not None else float('nan')
        print("lambda_raw: {:.5f} | lambda_eff: {:.5f} | grad: {:.3e} | loss: {:.5f}".format(
            lam_raw, lam_eff, lam_grad, float(loss_val)))
        traj_writer.writerow([epoch, idx, lam_raw, lam_eff, lam_grad, float(loss_val)])
        traj_fh.flush()

        if SAVE_DEBUG_VOLUMES:
            def _dump(vol, name):
                arr = np.array(vol[0, 0, :, :, :].detach().cpu())
                sitk.WriteImage(sitk.GetImageFromArray(arr),
                                './check_result/{}_{}_{}.nii.gz'.format(name, epoch, idx))
            _dump(source, 'source')
            _dump(src_lb, 'src_lbl')
            _dump(target, 'target')
            _dump(tar_lb, 'tar_lbl')
            _dump(x_aligned, 'rigid')

            velo = disp[0, ...].reshape(3, 96, 96, 96).permute(1, 2, 3, 0)
            velo = velo.detach().cpu().numpy()
            sitk.WriteImage(sitk.GetImageFromArray(velo, isVector=True),
                            './check_result/velo_im_{}_{}.nii.gz'.format(epoch, idx), False)

            velo = disp_lbl[0, ...].reshape(3, 96, 96, 96).permute(1, 2, 3, 0)
            velo = velo.detach().cpu().numpy()
            sitk.WriteImage(sitk.GetImageFromArray(velo, isVector=True),
                            './check_result/velo_lbl_{}_{}.nii.gz'.format(epoch, idx), False)

    print("training loss:", total)

    """ Save the trained model """
    if (epoch % SAVE_CHECKPOINT_EVERY == 0) or (epoch == para.solver.epochs - 1):
        checkpoint = {
            'epoch': epoch,
            'unimo_state_dict': net_obj.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lambda_raw': float(auto_weight.detach()),
            'lambda_eff': float(torch.clamp(auto_weight / LAMBDA_MAX, 0.0, 1.0).detach()),
            'lambda_max': LAMBDA_MAX,
        }
        model_name = "./saved_model/unimo_{}_epoch{}.pth".format(RUN_TAG, epoch)
        torch.save(checkpoint, model_name)

traj_fh.close()
print("lambda trajectory written to:", traj_file)
