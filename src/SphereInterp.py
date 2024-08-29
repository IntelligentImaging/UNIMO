
import torch
import torch.nn.functional as F


def rotation_matrix_to_quaternion(R):
    """ Convert a 3x3 rotation matrix to a quaternion. """
    R = R.squeeze(0)  # Remove batch dimension
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = (R[1,0] - R[0,1]) / s

    return torch.tensor([x, y, z, w], device=R.device)

def quaternion_to_rotation_matrix(q):
    """ Convert a quaternion to a 3x3 rotation matrix. """
    x, y, z, w = q
    R = torch.zeros((3, 3), device=q.device)
    R[0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[0, 1] = 2 * x * y - 2 * z * w
    R[0, 2] = 2 * x * z + 2 * y * w
    R[1, 0] = 2 * x * y + 2 * z * w
    R[1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[1, 2] = 2 * y * z - 2 * x * w
    R[2, 0] = 2 * x * z - 2 * y * w
    R[2, 1] = 2 * y * z + 2 * x * w
    R[2, 2] = 1 - 2 * x**2 - 2 * y**2
    return R

def slerp(q1, q2, t):
    """ Perform Spherical Linear Interpolation (SLERP) between two quaternions. """
    dot_product = torch.dot(q1, q2)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta_0 = torch.acos(dot_product)
    theta = theta_0 * t
    q2_prime = q2 - dot_product * q1
    q2_prime = q2_prime / torch.norm(q2_prime)
    q_interpolated = q1 * torch.cos(theta) + q2_prime * torch.sin(theta)
    return q_interpolated

def combine_rigid_transformations(xfm_I, xfm_G, lambda_param):
    # Ensure tensors are on the same device
    device = xfm_I.device
    assert xfm_I.device == xfm_G.device
    
    # Extract rotation and translation
    R_I = xfm_I[:, 0:3, 0:3]
    t_I = xfm_I[:, 0:3, 3:]
    R_G = xfm_G[:, 0:3, 0:3]
    t_G = xfm_G[:, 0:3, 3:]
    
    # Convert to quaternions
    q_I = rotation_matrix_to_quaternion(R_I)
    q_G = rotation_matrix_to_quaternion(R_G)
    
    # SLERP interpolation
    q_combined = slerp(q_I, q_G, lambda_param)
    
    # Convert back to rotation matrix
    R_combined = quaternion_to_rotation_matrix(q_combined)
    # Linear interpolation of translations
    t_combined = (1 - lambda_param) * t_I + lambda_param * t_G

    # Construct the final combined rigid transformation matrix
    Q_combined = torch.cat([R_combined.unsqueeze(0), t_combined], dim=2)
    return Q_combined

def gaussian_kernel(kernel_size, sigma, device):
    """Create a 3D Gaussian kernel."""
    kernel = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (kernel / sigma)**2)
    kernel = kernel / kernel.sum()  # Normalize to ensure the sum is 1
    kernel = kernel.unsqueeze(0)  # Add channel dimension
    kernel = kernel.unsqueeze(0)  # Add spatial dimensions
    kernel = kernel.expand(1, 1, kernel_size, kernel_size, kernel_size)  # Expand for 3D convolution

    return kernel

def gaussian_smoothing(binary_mask_gpu, kernel_size=3, sigma=.01):
    """Apply Gaussian smoothing to a 3D binary mask."""
    device = binary_mask_gpu.device
    
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma, device)
    
    # Apply Gaussian convolution
    smoothed_mask = F.conv3d(binary_mask_gpu, kernel, padding=kernel_size//2)
     # Normalize to [0, 1]
    smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min())
    
    return smoothed_mask