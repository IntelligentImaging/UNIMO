
import torch

##
## WARNING: translation is a proportion of image shape, to match the
##  affine_grid and grid_sample functions in torch

def rigid_transform_3D_PT(A,B,img_shape):
    '''
    This function assumes A, B are point matrices [B 3 K]
    '''

    #sub_from_all = torch.tensor(img_shape) / 2.0
    # assumes img_shape is torch tensor on correct device
    sub_from_all = img_shape / 2.0
    sub_from_all = torch.unsqueeze(sub_from_all,-1)
    sub_from_all = torch.unsqueeze(sub_from_all, 0)
    sub_from_all = torch.repeat_interleave( sub_from_all, A.size(0), dim=0 )
    sub_from_all = torch.repeat_interleave( sub_from_all, A.size(2), dim=2 )
    A = A - sub_from_all
    B = B - sub_from_all

    #assert A.shape == B.shape

    # find mean column wise
    centroid_A = torch.mean(A, axis=[2], keepdims=True)
    centroid_B = torch.mean(B, axis=[2], keepdims=True)

    # ensure centroids are 3x1
    #centroid_A = centroid_A.reshape(-1, 1)
    #centroid_B = centroid_B.reshape(-1, 1)

    centroid_A_block = torch.repeat_interleave(centroid_A,A.size(2),dim=2)
    centroid_B_block = torch.repeat_interleave(centroid_B,B.size(2),dim=2)

    # subtract mean
    Am = A - centroid_A_block
    Bm = B - centroid_B_block

    #H = Am @ tf.transpose(Bm)
    Bm = torch.transpose(Bm,1,2)

    H = torch.matmul( Am, Bm )

    # find rotation
    U, S, V = torch.svd(H, compute_uv=True)
    #Vt = torch.transpose(V,1,2)
    Ut = torch.transpose(U,1,2)
    R = torch.matmul(V, Ut)
    #Vt[...,2,:] *= 1

    # special reflection case
    #if tf.linalg.det(R) < 0:
    #Vt[...,2,:] *= -1
    dets = torch.det(R)
    dets = torch.unsqueeze(dets,-1)
    dets = torch.stack( [torch.ones_like(dets), torch.ones_like(dets), dets], axis=1 )
    dets = torch.cat( [dets,dets,dets], axis=2)

    V = V * torch.sign(dets)
    R = torch.matmul(V, Ut)

    #t = -R @ centroid_A + centroid_B
    t = ( torch.matmul(-R, centroid_A) + centroid_B ) 

    return R, t

##
## WARNING: translation is a proportion of image shape, to match the
##  affine_grid and grid_sample functions in torch

def rigid_transform_3D_PT_weighted(A,B,img_shape,w_A,w_B):
    '''
    This function assumes A, B are point matrices [B 3 K]
    and w_A and w_B are [B 1 K] non-negative weight arrays
    '''

    #sub_from_all = torch.tensor(img_shape) / 2.0
    # assumes img_shape is torch tensor on correct device
    sub_from_all = img_shape / 2.0
    sub_from_all = torch.unsqueeze(sub_from_all,-1)
    sub_from_all = torch.unsqueeze(sub_from_all, 0)
    sub_from_all = torch.repeat_interleave( sub_from_all, A.size(0), dim=0 )
    sub_from_all = torch.repeat_interleave( sub_from_all, A.size(2), dim=2 )
    A = A - sub_from_all
    B = B - sub_from_all

    #assert A.shape == B.shape

    #normalize weights
    w_A = w_A / w_A.sum(dim=2)
    w_B = w_B / w_B.sum(dim=2)

    # find weighed mean column wise
    centroid_A = torch.sum(A * w_A, axis=[2], keepdims=True)
    centroid_B = torch.sum(B * w_B, axis=[2], keepdims=True)

    # ensure centroids are 3x1
    #centroid_A = centroid_A.reshape(-1, 1)
    #centroid_B = centroid_B.reshape(-1, 1)

    centroid_A_block = torch.repeat_interleave(centroid_A,A.size(2),dim=2)
    centroid_B_block = torch.repeat_interleave(centroid_B,B.size(2),dim=2)

    # subtract mean
    Am = A - centroid_A_block
    Bm = B - centroid_B_block

    # at this point we shrink each {A,B}m matrix of vectors by their
    # respective weights
    Am = Am * w_A
    Bm = Bm * w_B

    #H = Am @ tf.transpose(Bm)
    Bm = torch.transpose(Bm,1,2)

    H = torch.matmul( Am, Bm )

    # find rotation
    U, S, V = torch.svd(H, compute_uv=True)
    #Vt = torch.transpose(V,1,2)
    Ut = torch.transpose(U,1,2)
    R = torch.matmul(V, Ut)
    #Vt[...,2,:] *= 1

    # special reflection case
    #if tf.linalg.det(R) < 0:
    #Vt[...,2,:] *= -1
    dets = torch.det(R)
    dets = torch.unsqueeze(dets,-1)
    dets = torch.stack( [torch.ones_like(dets), torch.ones_like(dets), dets], axis=1 )
    dets = torch.cat( [dets,dets,dets], axis=2)

    V = V * torch.sign(dets)
    R = torch.matmul(V, Ut)

    #t = -R @ centroid_A + centroid_B
    t = ( torch.matmul(-R, centroid_A) + centroid_B )

    return R, t



